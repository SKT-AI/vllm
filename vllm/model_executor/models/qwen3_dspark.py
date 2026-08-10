# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculative-decoding draft model -- CONFIDENCE variant.

This is the ``_conf`` A/B variant of the validated fixed-5 DSpark model
(``vllm_overlay_dspark/.../qwen3_dspark.py``). The ONLY functional difference
from the fixed-5 model is that this variant *loads* the trained confidence head
instead of dropping it, and exposes ``compute_confidence`` so the proposer can
gate the draft length by predicted per-position acceptance probability.

DSpark is DFlash plus a first-order Markov logit-bias head (``B = W1 @ W2``;
``W1`` indexes the verifier vocabulary, ``W2`` projects to the draft vocabulary).
On top of that, the training recipe adds a per-position *confidence head*: a
single ``nn.Linear(input_dim, 1)`` (see the training reference
``speculators/models/dspark/model_definitions.py::ConfidenceHead``) whose sigmoid
is the predicted probability that the drafted token at that block position will
be accepted by the verifier.

Confidence-head feature composition (verified against the training core.py
forward, ``confidence_head_with_markov=True``):

    conf_features[k] = concat( hidden_state[k],  markov_w1(prev_token[k]) )
    confidence[k]    = sigmoid( proj(conf_features[k]) )

where ``hidden_state[k]`` is the backbone hidden at block slot ``k`` and
``prev_token[k]`` is the same previous-block token id that seeds slot ``k``'s
Markov bias (anchor for k=0, then the token sampled at k-1). ``markov_w1`` is the
shared Markov W1 embedding (``prev_embeddings`` in training), so no extra
parameters are needed for the feature. With ``input_dim = hidden_size +
markov_rank = 7168 + 256 = 7424`` this matches the checkpoint's
``confidence_head.proj.weight [1, 7424]`` / ``confidence_head.proj.bias [1]``.

Everything else (backbone layers, ``fc``/``hidden_norm``/``norm``, the
cross-attention KV precompute, the Markov head, the draft->target vocab remap)
is inherited unchanged from :class:`DFlashQwen3ForCausalLM`.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

from .utils import AutoWeightsLoader, process_eagle_weight

logger = init_logger(__name__)


class ConfidenceHead(nn.Module):
    """Per-position acceptance-probability predictor (linear -> scalar logit).

    Mirrors ``speculators/models/dspark/model_definitions.py::ConfidenceHead``
    exactly (submodule name ``proj``) so the checkpoint keys
    ``confidence_head.proj.{weight,bias}`` load without renaming.
    """

    def __init__(self, input_dim: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, dtype=dtype)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


class DSparkQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """DFlash draft model + vanilla Markov bias + a loaded confidence head."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        cfg = self.config
        dflash_cfg = getattr(cfg, "dflash_config", {}) or {}
        self.markov_rank = int(
            getattr(cfg, "markov_rank", dflash_cfg.get("markov_rank", 0)) or 0
        )
        self.markov_head_type = getattr(
            cfg, "markov_head_type", dflash_cfg.get("markov_head_type", "vanilla")
        )
        if self.markov_rank <= 0:
            raise ValueError(
                "DSpark requires markov_rank > 0; got "
                f"{self.markov_rank}. A markov_rank of 0 is pure DFlash -- use "
                "method='dflash' instead."
            )
        if self.markov_head_type != "vanilla":
            raise NotImplementedError(
                "DSpark vLLM inference only supports markov_head_type='vanilla'; "
                f"got {self.markov_head_type!r}."
            )

        verifier_vocab_size = vllm_config.model_config.get_vocab_size()
        draft_vocab_size = self.config.draft_vocab_size
        params_dtype = vllm_config.model_config.dtype
        self.markov_w1 = nn.Embedding(
            verifier_vocab_size, self.markov_rank, dtype=params_dtype
        )
        self.markov_w2 = nn.Linear(
            self.markov_rank, draft_vocab_size, bias=False, dtype=params_dtype
        )

        # --- confidence head (the _conf variant's addition) -------------------
        # These flags default to the DSpark training recipe (enabled, and
        # conditioned on the Markov feature), so the head builds and loads even
        # when algos.py does not surface the flags into the hf_config. Override
        # via the draft hf_config or its dflash_config if a checkpoint differs.
        # UNVERIFIED ASSUMPTION: this variant targets a checkpoint that actually
        # contains confidence_head.proj weights (the one described in the task:
        # confidence_head_with_markov=True, enable_confidence_head=True). If the
        # checkpoint lacks them, load_weights would raise on the missing head;
        # build with enable_confidence_head=False for such checkpoints.
        self.enable_confidence_head = bool(
            getattr(
                cfg,
                "enable_confidence_head",
                dflash_cfg.get("enable_confidence_head", True),
            )
        )
        self.confidence_head_with_markov = bool(
            getattr(
                cfg,
                "confidence_head_with_markov",
                dflash_cfg.get("confidence_head_with_markov", True),
            )
        )
        self.confidence_head: ConfidenceHead | None = None
        if self.enable_confidence_head:
            hidden_size = int(self.config.hidden_size)
            conf_in = hidden_size + (
                self.markov_rank if self.confidence_head_with_markov else 0
            )
            self.confidence_head = ConfidenceHead(conf_in, dtype=params_dtype)
            logger.info(
                "DSpark(_conf): built confidence head input_dim=%d "
                "(hidden=%d + markov_rank=%d, with_markov=%s)",
                conf_in,
                hidden_size,
                self.markov_rank if self.confidence_head_with_markov else 0,
                self.confidence_head_with_markov,
            )

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Raw draft-vocabulary logits (before the draft->target expansion)."""
        return self.logits_processor(self.lm_head, hidden_states)

    def markov_bias(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        """Vanilla Markov bias for the previous (target-vocab) token ids."""
        prev_emb = self.markov_w1(prev_token_ids.long())
        return self.markov_w2(prev_emb)

    def compute_confidence(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Predicted per-position acceptance probability (sigmoid).

        Uses the SAME feature composition as training
        (``confidence_head_with_markov``): the backbone hidden state at the
        block slot concatenated with the Markov W1 embedding of the token that
        seeds that slot's Markov bias.

        Args:
            hidden_states: ``[B, H]`` backbone hidden states for one block slot.
            prev_token_ids: ``[B]`` verifier-vocab ids seeding this slot's Markov
                bias (anchor for slot 0, else the token sampled at the previous
                slot). Required when ``confidence_head_with_markov`` is set.

        Returns:
            ``[B]`` acceptance probabilities in ``[0, 1]``, or ``None`` if no
            confidence head is loaded.
        """
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert prev_token_ids is not None, (
                "confidence_head_with_markov requires prev_token_ids."
            )
            prev_emb = self.markov_w1(prev_token_ids.long()).to(hidden_states.dtype)
            feats = torch.cat([hidden_states, prev_emb], dim=-1)
        else:
            feats = hidden_states
        logit = self.confidence_head(feats.to(self.confidence_head.proj.weight.dtype))
        return torch.sigmoid(logit.float())

    def draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Map draft-vocab ids to target-vocab ids via the d2t offset table."""
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load DFlash backbone + Markov head + (this variant) the confidence head.

        Identical to the fixed-5 loader except ``confidence_head.*`` weights are
        routed to the top-level confidence head module instead of being dropped.
        """
        model_weights: dict[str, torch.Tensor] = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            if name.startswith("confidence_head."):
                # Keep the name as-is: self.confidence_head.proj.{weight,bias}
                # matches confidence_head.proj.{weight,bias} in the checkpoint.
                # Drop only if this variant was built without a confidence head.
                if self.confidence_head is not None:
                    model_weights[name] = loaded_weight
                continue
            if name.startswith("markov_head."):
                # markov_head.markov_w1.weight -> markov_w1.weight, etc.
                model_weights[name[len("markov_head.") :]] = loaded_weight
                continue

            assert "mask_hidden" not in name, (
                "DFlash should use mask_token_id to embed the padding hidden state"
            )
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs: list[str] = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        if self.confidence_head is None:
            skip_substrs.append("confidence_head")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
