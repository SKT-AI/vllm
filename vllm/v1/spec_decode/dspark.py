# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculative-decoding proposer -- CONFIDENCE variant.

This is the ``_conf`` A/B variant of the validated fixed-5 DSpark proposer
(``vllm_overlay_dspark/.../dspark.py``). It keeps the fixed-5 proposer's Markov
sampling and the ``sample_from_anchor`` realignment fix VERBATIM, and adds ONE
thing: confidence-based dynamic draft length.

While sampling block position ``k`` left-to-right, the loaded confidence head
(see ``qwen3_dspark.py`` in this variant) predicts the acceptance probability of
the drafted token at ``k`` (sigmoid). After the block is sampled, each request's
draft is truncated to the confident prefix: the draft is cut at the first
position ``k`` whose confidence falls below ``DSPARK_CONF_THRESHOLD`` (default
0.3), emitting only positions ``0..k-1``.

RETURN PATH -- true variable length vs. rectangular padding
-----------------------------------------------------------
The base ``propose()`` (llm_base_proposer.py) returns a RECTANGULAR
``[B, num_speculative_tokens]`` tensor for the parallel-drafting path
(``draft_token_ids.view(-1, num_speculative_tokens)``). However, the rest of the
draft-token plumbing DOES accept ragged per-request lists:

  * gpu_model_runner ``_get_draft_token_ids_cpu`` / ``take_draft_token_ids``
    special-case ``isinstance(self._draft_token_ids, list)`` and return it
    as-is (this is the path ngram / suffix / medusa already use);
  * scheduler ``update_draft_token_ids`` stores ``request.spec_token_ids =
    spec_token_ids`` verbatim (no re-padding, no -1 stripping);
  * the async scheduler sizes each request's block from
    ``len(spec_decode_tokens.get(req_id, ()))`` and the verifier consumes
    ``num_draft_tokens = len(scheduled_spec_token_ids)``.

So returning a ragged ``list[list[int]]`` of truncated drafts yields REAL
verifier-compute savings (fewer draft tokens verified for low-confidence
requests) WITHOUT any scheduler/kernel change. This is the DEFAULT here.

CAVEAT (UNVERIFIED, documented): the async-scheduling D2H copy path
``_copy_draft_token_ids_to_cpu`` only runs for structured-output / penalties /
bad_words requests and it early-returns when ``self._draft_token_ids`` is not a
tensor. For those request kinds a ragged-list return would skip that copy (the
plain acceptance test uses none of them, so it is unaffected). For that case a
rectangular fallback is provided via ``DSPARK_CONF_RECTANGULAR=1`` -- it keeps
the ``[B, num_spec]`` tensor and overwrites post-cutoff positions with the last
confident token id. THAT FALLBACK YIELDS NO VERIFIER-COMPUTE SAVINGS (the
verifier still processes all ``num_spec`` positions); it only caps effective
acceptance. Prefer the default ragged path.
"""

import os

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer

logger = init_logger(__name__)


class DSparkProposer(DFlashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        # Per-request anchor token (last verified token id, target vocab) used
        # to seed the Markov chain at block position 0. Refreshed every call to
        # set_inputs_first_pass.
        self._dspark_anchor_ids: torch.Tensor | None = None
        # Per-position confidence [B, num_spec] stashed by _sample_draft_tokens
        # and consumed by propose() for truncation.
        self._last_conf: torch.Tensor | None = None
        # DSpark checkpoints are trained with sample_from_anchor=True (the anchor
        # slot itself predicts the first speculative token). The container's
        # DFlash path hard-codes the sample_from_anchor=False convention, so we
        # realign the sampled slots (see set_inputs_first_pass).
        self.sample_from_anchor = bool(
            self.dflash_config.get("sample_from_anchor", True)
        )
        # Confidence threshold: draft is truncated at the first block position
        # whose predicted acceptance probability is below this value. <=0
        # disables truncation (falls back to fixed-num_spec, i.e. fixed-5).
        self.conf_threshold = float(os.environ.get("DSPARK_CONF_THRESHOLD", "0.3"))
        # Rectangular sentinel fallback (no verifier savings) -- see module docstring.
        self.conf_rectangular = os.environ.get("DSPARK_CONF_RECTANGULAR", "0") == "1"
        logger.info(
            "DSpark(_conf) proposer: DSPARK_CONF_THRESHOLD=%.4f rectangular=%s",
            self.conf_threshold,
            self.conf_rectangular,
        )

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        # next_token_ids holds the last verified token per request -- the anchor
        # for block position 0's Markov bias.
        self._dspark_anchor_ids = next_token_ids
        num_tokens, tis, new_cad = super().set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
        if self.sample_from_anchor:
            # --- sample_from_anchor realignment (the DSpark correctness fix) ---
            # See the fixed-5 overlay for the full derivation. The mask-slot
            # offsets are [req*(N+1)+1 .. req*(N+1)+N]; subtracting 1 selects the
            # anchor slot plus the first N-1 masks, matching the trained
            # sample_from_anchor=True layout exactly.
            tis = tis - 1
        return num_tokens, tis, new_cad

    @override
    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample the block left-to-right with the Markov bias.

        Identical to the fixed-5 sampler, plus: for each block position ``k`` it
        also evaluates the confidence head (using the SAME hidden state and
        previous-token as the Markov bias) and stashes the per-position
        acceptance probabilities in ``self._last_conf`` ([B, num_spec]) for
        ``propose()`` to truncate on.
        """
        num_spec = self.num_speculative_tokens
        total = hidden_states.shape[0]
        batch_size = total // num_spec
        assert batch_size * num_spec == total, (
            "DSpark expected hidden_states with a multiple of "
            f"num_speculative_tokens={num_spec} rows, got {total}."
        )
        h = hidden_states.view(batch_size, num_spec, -1)

        anchor = self._dspark_anchor_ids
        assert anchor is not None, (
            "DSpark anchor ids are unset; set_inputs_first_pass must run first."
        )
        prev = anchor[:batch_size].to(device=hidden_states.device).long()

        out_cols: list[torch.Tensor] = []
        conf_cols: list[torch.Tensor] = []
        want_conf = self.conf_threshold > 0.0 and getattr(
            self.model, "confidence_head", None
        ) is not None
        for k in range(num_spec):
            # Confidence for slot k uses the SAME (hidden_k, prev_token_k) as the
            # Markov bias below -- evaluate it BEFORE sampling this slot so
            # ``prev`` still holds slot k's conditioning token.
            if want_conf:
                conf_k = self.model.compute_confidence(h[:, k, :], prev)
                if conf_k is not None:
                    conf_cols.append(conf_k)

            # Draft-vocab logits for this block position (transformer already
            # ran; this is just the lm_head projection).
            draft_logits = self.model.compute_draft_logits(h[:, k, :])
            bias = self.model.markov_bias(prev).to(draft_logits.dtype)
            draft_ids = (draft_logits + bias).argmax(dim=-1)
            # Map the draft-vocab id back to the target vocabulary and use it as
            # the "previous token" that conditions the next block position.
            target_ids = self.model.draft_to_target(draft_ids)
            out_cols.append(target_ids)
            prev = target_ids

        self._last_conf = (
            torch.stack(conf_cols, dim=1) if len(conf_cols) == num_spec else None
        )

        # Re-assemble batch-major flat [B * num_spec] to match the token
        # ordering expected by propose().
        draft_token_ids = torch.stack(out_cols, dim=1).reshape(-1).int()
        return draft_token_ids, None

    def _compute_cutoffs(self, conf: torch.Tensor) -> torch.Tensor:
        """First block position below threshold per request (else num_spec).

        Args:
            conf: ``[B, num_spec]`` per-position acceptance probabilities.

        Returns:
            ``[B]`` int64 valid lengths in ``[0, num_spec]``.
        """
        num_spec = conf.shape[1]
        below = conf < self.conf_threshold  # [B, num_spec]
        # argmax over a bool row returns the first True index, or 0 if all False.
        first_below = below.int().argmax(dim=1)
        any_below = below.any(dim=1)
        cutoffs = torch.where(
            any_below,
            first_below,
            torch.full_like(first_below, num_spec),
        )
        return cutoffs.long()

    @override
    def propose(self, *args, **kwargs):
        """Sample the block, then truncate each request to its confident prefix.

        Returns a ragged ``list[list[int]]`` (true variable length -> real
        verifier savings) by default, or a rectangular ``[B, num_spec]`` tensor
        with sentinel padding (no savings) when ``DSPARK_CONF_RECTANGULAR=1``.
        Falls back to the base rectangular tensor unchanged when truncation is
        disabled or unavailable.
        """
        draft = super().propose(*args, **kwargs)  # [B, num_spec] tensor

        # Truncation disabled, or not the parallel-drafting rectangular path we
        # know how to truncate -> behave exactly like fixed-5.
        if self.conf_threshold <= 0.0 or not isinstance(draft, torch.Tensor):
            return draft
        if draft.dim() != 2:
            return draft
        conf = self._last_conf
        self._last_conf = None
        if conf is None or conf.shape != draft.shape:
            if conf is not None:
                logger.warning_once(
                    "DSpark(_conf): confidence shape %s != draft shape %s; "
                    "skipping truncation.",
                    tuple(conf.shape),
                    tuple(draft.shape),
                )
            return draft

        cutoffs = self._compute_cutoffs(conf)  # [B]

        if self.conf_rectangular:
            # Rectangular sentinel fallback: overwrite post-cutoff positions with
            # the last confident token id (a benign, guaranteed-rejected repeat
            # once the prefix diverges). NO verifier-compute savings: the
            # verifier still processes all num_spec positions.
            num_spec = draft.shape[1]
            col = torch.arange(num_spec, device=draft.device).unsqueeze(0)
            keep = col < cutoffs.unsqueeze(1)  # [B, num_spec]
            # index of last kept position (>=0); if cutoff==0, use position 0.
            last_kept = (cutoffs - 1).clamp(min=0)
            fill = draft.gather(1, last_kept.unsqueeze(1)).expand_as(draft)
            return torch.where(keep, draft, fill)

        # Default: true variable-length ragged list -> real verifier savings.
        draft_cpu = draft.tolist()
        cut_cpu = cutoffs.tolist()
        return [row[:c] for row, c in zip(draft_cpu, cut_cpu)]
