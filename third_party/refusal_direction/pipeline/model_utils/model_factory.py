# Modified from the original: https://github.com/andyrdt/refusal_direction (Apache-2.0).
# See third_party/refusal_direction/UPSTREAM.md for the full list of changes.

from pipeline.model_utils.model_base import ModelBase

def construct_model_base(model_path: str) -> ModelBase:

    # Wrappers exist only for the two families the released refusal directions
    # were extracted for: Qwen2.5 and Llama-3.  Anything else raises.
    if 'qwen' in model_path.lower():
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)
    if 'llama-3' in model_path.lower():
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
