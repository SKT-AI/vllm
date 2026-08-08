# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculative-decoding draft model.

DSpark is DFlash plus a first-order Markov logit-bias head: after the DFlash
backbone produces per-position draft logits, position ``k`` is biased by a
low-rank function of the previously sampled token in the block
(``B = W1 @ W2``; ``W1`` indexes the verifier vocabulary, ``W2`` projects to
the draft vocabulary). This mirrors the training-side reference in
``speculators/models/dspark`` (``MarkovHead`` with ``markov_head_type="vanilla"``).

The training-time confidence head is intentionally *not* loaded here: vLLM's
rejection sampler decides acceptance from the draft/target distributions, so the
confidence head plays no role at inference.

Everything else (backbone layers, ``fc``/``hidden_norm``/``norm``, the
cross-attention KV precompute, the draft->target vocab remap) is inherited
unchanged from :class:`DFlashQwen3ForCausalLM`.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

from .utils import AutoWeightsLoader, process_eagle_weight

logger = init_logger(__name__)


class DSparkQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """DFlash draft model + a vanilla (first-order) Markov logit-bias head."""

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
            # 'gated'/'rnn' additionally consume the backbone hidden state (and,
            # for 'rnn', a recurrent state) per block position. Only the vanilla
            # first-order bias is wired for vLLM inference.
            raise NotImplementedError(
                "DSpark vLLM inference only supports markov_head_type='vanilla'; "
                f"got {self.markov_head_type!r}."
            )

        # W1 indexes the verifier (target) vocabulary by the previous token id;
        # W2 projects the rank-r embedding to the draft vocabulary so the bias
        # adds directly onto the DFlash draft logits.
        verifier_vocab_size = vllm_config.model_config.get_vocab_size()
        draft_vocab_size = self.config.draft_vocab_size
        params_dtype = vllm_config.model_config.dtype
        # Plain (fully replicated) modules: the head is tiny relative to the
        # backbone and this avoids TP-sharding bookkeeping. Names match the
        # checkpoint's ``markov_head.markov_w1`` / ``markov_head.markov_w2``
        # after the prefix strip in load_weights().
        self.markov_w1 = nn.Embedding(
            verifier_vocab_size, self.markov_rank, dtype=params_dtype
        )
        self.markov_w2 = nn.Linear(
            self.markov_rank, draft_vocab_size, bias=False, dtype=params_dtype
        )

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Raw draft-vocabulary logits (before the draft->target expansion).

        The Markov bias lives in draft-vocabulary space, so DSpark adds it here
        rather than to the target-expanded logits returned by ``compute_logits``.
        """
        return self.logits_processor(self.lm_head, hidden_states)

    def markov_bias(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        """Vanilla Markov bias for the previous (target-vocab) token ids.

        Args:
            prev_token_ids: ``[B]`` token ids in the verifier/target vocabulary.

        Returns:
            ``[B, draft_vocab_size]`` additive logit bias.
        """
        prev_emb = self.markov_w1(prev_token_ids.long())
        return self.markov_w2(prev_emb)

    def draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Map draft-vocab ids to target-vocab ids via the d2t offset table."""
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load DFlash backbone weights plus the Markov head.

        Mirrors :meth:`DFlashQwen3ForCausalLM.load_weights` but additionally
        routes ``markov_head.markov_w{1,2}.weight`` to the top-level Markov
        modules and drops the (inference-unused) confidence head.
        """
        model_weights: dict[str, torch.Tensor] = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            # The confidence head is a training-time auxiliary; vLLM's rejection
            # sampler decides acceptance, so it is not needed at inference.
            if "confidence_head" in name:
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
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
