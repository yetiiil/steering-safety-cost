"""Generate reference continuations with steering on every forward pass.

Reference generation uses the underlying HuggingFace model. A forward hook
applies the intervention during both prefill and cached decoding, whereas a
single NNsight 0.3.x output assignment targets only one module invocation.
Returned masks distinguish continuation tokens (including EOS) from padding.
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def steering_hook(model, layer, vector, multiplier):
    """Apply an additive residual intervention at every token and remove it on exit."""
    if vector is None or multiplier == 0:
        yield
        return
    if layer is None:
        raise ValueError("A steering layer is required with a steering vector")

    def add_vector(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        delta = multiplier * vector.to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden + delta
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    handle = model.model.layers[layer].register_forward_hook(add_vector)
    try:
        yield
    finally:
        handle.remove()


def generate_reference_continuations(
    model, tokenizer, prompts, *, max_new_tokens, batch_size,
    layer=None, vector=None, multiplier=0.0,
):
    """Return CPU IDs and masks for globally left-padded reference trajectories.

    The final max_new_tokens positions are reserved for the continuation.
    Right-padding after early EOS is excluded by both attention_mask and
    continuation_mask. No prompt/header token is included in the continuation.
    """
    if not prompts or max_new_tokens < 1 or batch_size < 1:
        raise ValueError("Nonempty prompts and positive token/batch sizes are required")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("A padding token is required for reference generation")
    device = next(model.parameters()).device
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_ids = eos_ids or []
    batches = []
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        with torch.no_grad(), steering_hook(model, layer, vector, multiplier):
            for start in range(0, len(prompts), batch_size):
                inputs = tokenizer(prompts[start:start + batch_size], padding=True,
                                   return_tensors="pt").to(device)
                prompt_width = inputs.input_ids.shape[1]
                generated = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None, pad_token_id=pad_id,
                )
                continuation = generated[:, prompt_width:]
                valid = torch.ones_like(continuation, dtype=torch.bool)
                for eos_id in eos_ids:
                    # Include the first EOS itself, excluding all later positions.
                    eos = continuation.eq(eos_id)
                    valid &= (eos.cumsum(-1) - eos.long()).eq(0)
                missing = max_new_tokens - continuation.shape[1]
                if missing < 0:
                    raise ValueError("Model generated more than max_new_tokens")
                generated = F.pad(generated, (0, missing), value=pad_id)
                valid = F.pad(valid, (0, missing), value=False)
                attention = torch.cat([inputs.attention_mask, valid.long()], dim=1)
                batches.append((generated.cpu(), attention.cpu(), valid.cpu()))
    finally:
        tokenizer.padding_side = padding_side

    width = max(ids.shape[1] for ids, _, _ in batches)
    return {
        "input_ids": torch.cat([F.pad(ids, (width - ids.shape[1], 0), value=pad_id)
                                for ids, _, _ in batches]),
        "attention_mask": torch.cat([F.pad(mask, (width - mask.shape[1], 0), value=0)
                                     for _, mask, _ in batches]),
        "continuation_mask": torch.cat([mask for _, _, mask in batches]),
    }
