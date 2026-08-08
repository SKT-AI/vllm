# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

SUPPORTED_SPECULATORS_TYPES = {}


def register_speculator(name):
    def decorator(fn):
        SUPPORTED_SPECULATORS_TYPES[name] = fn
        return fn

    return decorator


@register_speculator("eagle3")
def update_eagle3(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply Eagle-3 specific configuration transformations to the `dict` used to
    construct the Transformers PreTrainedConfig.

    Eagle-3 specific fields:
    - draft_vocab_size: Size of the draft model's vocabulary
    - target_hidden_size: Hidden size of the target model
    - norm_before_residual: Whether to apply norm before residual connection
    - norm_before_fc: Whether to apply RMSNorm before the fc projection
    - eagle_aux_hidden_state_layer_ids: List of layer indices from the base
        model to use as auxiliary inputs for the Eagle3 drafter. These layers
        provide intermediate hidden states that help the drafter make better
        predictions. This is the standard field used in Eagle3 checkpoints.
    """

    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        pre_trained_config["target_hidden_size"] = config_dict["target_hidden_size"]
    pre_trained_config["norm_before_residual"] = config_dict.get(
        "norm_before_residual", True
    )
    pre_trained_config["norm_before_fc"] = config_dict.get("norm_before_fc", False)
    pre_trained_config["architectures"] = ["Eagle3LlamaForCausalLM"]
    if config_dict.get("eagle_aux_hidden_state_layer_ids"):
        pre_trained_config["eagle_aux_hidden_state_layer_ids"] = config_dict[
            "eagle_aux_hidden_state_layer_ids"
        ]


@register_speculator("peagle")
def update_peagle(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply PEagle (Parallel Eagle) specific configuration transformations to
    the `dict` used to construct the Transformers PreTrainedConfig.

    PEagle specific fields:
    - draft_vocab_size: Size of the draft model's vocabulary
    - target_hidden_size: Hidden size of the target model
    - norm_before_residual: Whether to apply norm before residual connection
    - norm_before_fc: Whether to apply RMSNorm before the fc projection
    - mask_token_id (required): Token ID used for parallel drafting mask
        placeholders, mapped to pard_token for the proposer
    - eagle_aux_hidden_state_layer_ids: Layer indices from the target model
        whose intermediate hidden states are used as auxiliary inputs
    """
    pre_trained_config["architectures"] = ["PeagleLlamaForCausalLM"]
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        pre_trained_config["target_hidden_size"] = config_dict["target_hidden_size"]
    pre_trained_config["norm_before_residual"] = config_dict.get(
        "norm_before_residual", False
    )
    pre_trained_config["norm_before_fc"] = config_dict.get("norm_before_fc", False)
    pre_trained_config["pard_token"] = config_dict["mask_token_id"]
    if config_dict.get("eagle_aux_hidden_state_layer_ids"):
        pre_trained_config["eagle_aux_hidden_state_layer_ids"] = config_dict[
            "eagle_aux_hidden_state_layer_ids"
        ]


@register_speculator("dflash")
def update_dflash(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply DFlash specific configuration transformations to the `dict` used to
    construct the Transformers PreTrainedConfig.

    DFlash specific fields:
    - draft_vocab_size: Size of the draft model's vocabulary
    - target_hidden_size: Hidden size of the target model
    - mask_token_id (required): Token ID used for parallel drafting mask
        placeholders
    - aux_hidden_state_layer_ids (required): Layer indices from the target
        model whose intermediate hidden states are used as context for the
        DFlash drafter. Mapped to both eagle_aux_hidden_state_layer_ids
        (for gpu_model_runner) and dflash_config.target_layer_ids (for the
        DFlash model).
    """
    pre_trained_config["architectures"] = ["DFlashDraftModel"]
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        pre_trained_config["target_hidden_size"] = config_dict["target_hidden_size"]

    aux_layer_ids = config_dict["aux_hidden_state_layer_ids"]
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids

    # DFlash configs use different indexing for the target layers, see #40727
    pre_trained_config["dflash_config"] = {
        "mask_token_id": config_dict["mask_token_id"],
        "target_layer_ids": [i - 1 for i in aux_layer_ids],
    }


@register_speculator("dspark")
def update_dspark(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply DSpark specific configuration transformations to the `dict` used to
    construct the Transformers PreTrainedConfig.

    DSpark is DFlash plus a first-order Markov logit-bias head (and a
    training-time confidence head that is unused at inference). It shares every
    DFlash field, so we reuse ``update_dflash`` and then:
    - override ``architectures`` to load the DSpark model class (which adds the
      Markov head weights on top of the DFlash backbone), and
    - surface the Markov head hyper-parameters (``markov_rank``,
      ``markov_head_type``) so the DSpark model can build the head.

    DSpark specific fields:
    - markov_rank: low-rank dim of the Markov bias factorization B = W1 @ W2.
    - markov_head_type: Markov head variant. Only "vanilla" (first-order,
      previous-token-only) is supported at inference.
    """
    update_dflash(config_dict=config_dict, pre_trained_config=pre_trained_config)
    pre_trained_config["architectures"] = ["DSparkDraftModel"]

    markov_rank = config_dict.get("markov_rank", 256)
    markov_head_type = config_dict.get("markov_head_type", "vanilla")
    pre_trained_config["markov_rank"] = markov_rank
    pre_trained_config["markov_head_type"] = markov_head_type
    # Also mirror into dflash_config so anything reading the drafter sub-config
    # (see DFlashQwen3Model) can find the Markov settings.
    pre_trained_config["dflash_config"]["markov_rank"] = markov_rank
    pre_trained_config["dflash_config"]["markov_head_type"] = markov_head_type

    # sample_from_anchor controls the block layout the draft was trained for.
    # DSpark defaults to True (the anchor slot predicts the first speculative
    # token). The DSparkProposer reads this to realign the sampled query slots
    # against the container's DFlash (sample_from_anchor=False) convention.
    pre_trained_config["dflash_config"]["sample_from_anchor"] = config_dict.get(
        "sample_from_anchor", True
    )
