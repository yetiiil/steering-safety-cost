"""Chat templates used when building prompts for optimisation.

The templates are written out literally rather than taken from
`tokenizer.apply_chat_template` so that the prompt strings are identical to the
ones used to produce the released vectors, independent of the `transformers`
version installed.
"""

QWEN25_CHAT_TEMPLATE = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def apply_chat_template(model_path: str, instructions: list[str]) -> list[str]:
    """Wrap each instruction in the chat template of `model_path`'s family."""
    key = model_path.lower()
    if "llama-3" in key:
        template = LLAMA3_CHAT_TEMPLATE
    elif "qwen2.5" in key:
        template = QWEN25_CHAT_TEMPLATE
    else:
        raise ValueError(
            f"Model {model_path} not supported: add its chat template to "
            "cast.chat_templates."
        )
    return [template.format(instruction=inst) for inst in instructions]
