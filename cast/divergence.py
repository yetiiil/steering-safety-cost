"""Divergence terms and refusal-token sets used by the CAST objective.

`masked_kl_div_fn` implements D_R of Eq. (3): the refusal-token slice of the
forward KL, with both distributions obtained by softmax over the *full*
vocabulary (the slice is taken after the softmax, so the terms are not
renormalised over R).  It is the safety objective (Eq. 4) and the false-refusal
constraint (Eq. 6); `kl_div_fn` is the full-vocabulary KL of the effect
constraint (Eq. 5).
"""

import torch

# Full refusal-indicative token sets used by the research implementation.
# Table 7 omits additional refusal cues, including initial "I" (40) and the
# Qwen IDs for "provide"/"request". Retain the complete research vocabulary.
# Each ID occurs once; the historical duplicate 47856 has no set semantics.
REFUSAL_TOKENS = {
    "qwen2.5": [
        40, 72, 8088, 76, 83, 359, 67597, 1921, 2121, 75998, 33260,
        74842, 64837, 61456, 47856, 4814, 50966, 2035, 7995, 2685,
    ],
    "llama-3": [
        40, 72, 83, 4919, 1285, 7712, 2079, 9210, 34360, 1962,
        62556, 24343, 685, 26125,
    ],
}


def get_refusal_tokens(model_path: str) -> list[int]:
    """Return the refusal-token ids for `model_path` (a HuggingFace model id)."""
    key = model_path.lower()
    for family, tokens in REFUSAL_TOKENS.items():
        if family in key:
            return list(tokens)
    raise ValueError(
        f"Model {model_path} not supported: add its refusal tokens to "
        "cast.divergence.REFUSAL_TOKENS."
    )


def _masked_position_mean(values, position_mask, reduction):
    """Average valid continuation positions per prompt, then over prompts.

    position_mask is a concrete tensor on the same device as the logits.
    Keeping its shape concrete also permits use inside an NNsight trace.
    """
    mask = position_mask.to(torch.float32)
    weighted = values.reshape(mask.shape) * mask
    if reduction == "sum":
        return weighted.sum()
    return (weighted.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)).mean()


def kl_div_fn(logits_a, logits_b, reduction: str = "batchmean", position_mask=None):
    """Full-vocabulary KL(a ‖ b), where b is the reference distribution.

    Args:
        logits_a: logits of the distribution divergence is measured *from*.
        logits_b: logits of the reference distribution.
        reduction: passed through to torch.nn.functional.kl_div.
    """
    logits_a = logits_a.to(torch.float32)
    logits_b = logits_b.to(torch.float32)

    # torch's kl_div(log_p, q) computes sum(q · (log q − log_p)) = KL(q ‖ p),
    # so the arguments are passed b-then-a to obtain KL(a ‖ b).
    terms = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits_b, dim=-1),
        torch.nn.functional.softmax(logits_a, dim=-1),
        reduction="none" if position_mask is not None else reduction,
    )
    if position_mask is not None:
        return _masked_position_mean(terms.sum(dim=-1), position_mask, reduction)
    return terms


def masked_kl_div_fn(
    logits_a,
    logits_b,
    token_indices,
    reduction: str = "mean",
    position_mask=None,
):
    """Refusal-token divergence D_R of Eq. (3).

        D_R(b ‖ a) = Σ_{x ∈ token_indices} P_b(x) · log( P_b(x) / P_a(x) )

    with b the fixed baseline and a the model being optimised.  Weighting by
    P_b means the term is dominated by refusal tokens the baseline would
    actually emit.  The value can be negative when a places *more* mass on
    refusal tokens than b — which is why minimising the safety loss can push
    refusal above the unsteered baseline rather than merely restoring it.

    Args:
        logits_a: [T, V] or [B, T, V] logits of the model being optimised.
        logits_b: [T, V] or [B, T, V] logits of the fixed baseline.
        token_indices: vocab indices to sum over (see REFUSAL_TOKENS).
        reduction: 'mean' (default) or 'sum' over time steps.
    """
    logits_a = logits_a.to(torch.float32)
    logits_b = logits_b.to(torch.float32)

    # Flatten any leading batch dimension: [B, T, V] -> [B*T, V], no-op for [T, V].
    # The reshape is unconditional to avoid a Python bool test on an nnsight Proxy.
    logits_a = logits_a.reshape(-1, logits_a.shape[-1])
    logits_b = logits_b.reshape(-1, logits_b.shape[-1])

    # Full-vocabulary log-softmax, then slice to the refusal tokens.
    log_pa = torch.nn.functional.log_softmax(logits_a, dim=-1)  # [T, V]
    log_pb = torch.nn.functional.log_softmax(logits_b, dim=-1)  # [T, V]

    log_pa_s = log_pa[:, token_indices]
    log_pb_s = log_pb[:, token_indices]

    p_b = torch.nn.functional.softmax(logits_b, dim=-1)  # [T, V]
    p_b_s = p_b[:, token_indices]

    per_step_kl = (p_b_s * (log_pb_s - log_pa_s)).sum(dim=-1)

    if position_mask is not None:
        return _masked_position_mean(per_step_kl, position_mask, reduction)

    return per_step_kl.mean() if reduction == "mean" else per_step_kl.sum()
