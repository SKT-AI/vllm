# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculative-decoding proposer.

DSpark = DFlash backbone + a first-order Markov logit-bias applied while
sampling the drafted block. DFlash proposes the whole block in a single
parallel forward pass and then takes an independent argmax per position
(:meth:`SpecDecodeBaseProposer._sample_draft_tokens`). DSpark keeps that single
forward pass but replaces the independent argmax with a short left-to-right
sweep over the block positions: position ``k`` is biased by a low-rank function
of the token actually sampled at position ``k-1`` (position 0 is seeded by the
last verified token, i.e. the anchor). The bias is a cheap
``[B, rank] @ [rank, draft_vocab]`` GEMM per position -- the transformer is
*not* re-run, so the extra cost is negligible.

The training-time confidence head is not consumed here (vLLM's rejection
sampler decides acceptance).
"""

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
        # DSpark checkpoints are trained with sample_from_anchor=True (the anchor
        # slot itself predicts the first speculative token). The container's
        # DFlash path hard-codes the sample_from_anchor=False convention, so we
        # realign the sampled slots (see set_inputs_first_pass).
        self.sample_from_anchor = bool(
            self.dflash_config.get("sample_from_anchor", True)
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
            # DFlash builds num_speculative_tokens+1 query slots per request:
            #   query_off 0 = the bonus/anchor slot, carrying the REAL anchor
            #                 token (next_token_id) at the anchor position;
            #   query_off 1..N = N mask ("in-fill") slots.
            # The container's kernel writes token_indices_to_sample = the N mask
            # slots (offsets 1..N). That is the sample_from_anchor=FALSE
            # convention: a mask at position q predicts token(q).
            #
            # But this checkpoint was trained sample_from_anchor=TRUE: the anchor
            # slot itself predicts the first speculative token, slot i sits at
            # position anchor+i and predicts token(anchor+i+1), and there are
            # only N-1 mask slots. Sampling offsets 1..N therefore reads every
            # slot one position too late -> a systematic off-by-one that
            # collapses acceptance (monotonic but ~3x below the trained rate).
            #
            # The mask-slot offsets are exactly [req*(N+1)+1 .. req*(N+1)+N];
            # subtracting 1 selects [req*(N+1)+0 .. req*(N+1)+(N-1)] = the anchor
            # slot plus the first N-1 masks. Their positions, input embeddings
            # (real anchor token at slot 0, masks after) and per-slot targets
            # then match the trained sample_from_anchor=True layout exactly, and
            # the Markov seed (anchor token -> slot 0) lines up with training.
            #
            # NOTE: DFlash still builds N+1 query slots, so a single unused mask
            # (offset N) remains in the batch and is visible to the non-causal
            # full-attention layers. Its effect is tiny (one extra block key
            # among the full verifier-context K/V) but nonzero; if the re-test
            # still trails the trained acceptance, the residual fix is to build
            # exactly N query slots for sample_from_anchor (kernel change).
            tis = tis - 1
        return num_tokens, tis, new_cad

    @override
    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample the block left-to-right with the Markov bias.

        ``hidden_states`` are the backbone outputs for every block slot, in
        batch-major order ``[b0_s0, b0_s1, ..., b1_s0, ...]`` (shape
        ``[B * num_speculative_tokens, H]``), matching how ``propose`` reshapes
        the result with ``.view(-1, num_speculative_tokens)``.
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
        for k in range(num_spec):
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

        # Re-assemble batch-major flat [B * num_spec] to match the token
        # ordering expected by propose().
        draft_token_ids = torch.stack(out_cols, dim=1).reshape(-1).int()
        return draft_token_ids, None
