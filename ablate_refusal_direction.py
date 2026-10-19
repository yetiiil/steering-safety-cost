"""
Refusal Direction Ablation -- the "Refusal Direction Ablation" baseline of Table 1.

The baseline asks the obvious question CAST is compared against: if the safety
cost of a steering vector came from its overlap with the model's refusal
direction, does simply projecting that direction out fix it? This script builds
those vectors. It takes the normalized CAA vectors in artifacts/caa_vectors/,
removes the DIM refusal direction of the corresponding model, rescales to the
original norm, and writes artifacts/refusal_ablated_vectors/. The evaluation
scripts then treat the result like any other steering vector.

Note what is *not* happening here: the direction is fixed, taken from the DIM
pipeline, and never sees the steering vector it is removed from. That is the
whole contrast with CAST, which optimises its direction under the effect and
false-refusal constraints.

The baseline covers the three models the paper reports -- Llama-3.1-8B-Instruct,
Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct. Adding a model means adding its
filename pattern to `infer_refusal_subdir_from_filename` and placing its
direction under artifacts/DIM/<model_id>/direction.pt.
"""

import os
import argparse
from typing import Optional, Tuple, List

import torch

from cast.paths import (CAA_VECTOR_DIR, REFUSAL_ABLATED_VECTOR_DIR,
                        dim_direction_dir, require)


# ============================================================================
# Constants
# ============================================================================

# The only ablation the paper reports. Kept as a list so that --ablation_type
# stays a documented choice rather than an unexplained constant.
ABLATION_TYPES = ["direction"]
SUPPORTED_FILE_EXTENSION = ".pt"


# ============================================================================
# Vector Loading and Processing
# ============================================================================

def infer_refusal_subdir_from_filename(filename: str) -> Optional[str]:
    """Infer the model id from a normalized CAA vector filename.

    generate_vectors.py encodes the HuggingFace model id in the filename with
    `/` and `-` replaced by `_`, so the patterns below are matched against that
    mangled form (and against a couple of variants that older vector files use).

    Only the three models the paper reports are recognised; anything else
    returns None and is reported as an error by the caller rather than silently
    ablated with the wrong direction.
    """
    name = os.path.basename(filename)
    lower_name = name.lower()

    if "llama_3" in lower_name:
        return "llama-3.1-8b-instruct"
    if "qwen2.5_7b" in lower_name or "qwen_2.5_7b" in lower_name or "qwen2_5_7b" in lower_name:
        return "qwen2.5-7b-instruct"
    if "qwen2.5_14b" in lower_name or "qwen_2.5_14b" in lower_name or "qwen2_5_14b" in lower_name:
        return "qwen2.5-14b-instruct"

    return None


def load_refusal_direction(
    refusal_root: Optional[str],
    subdir: str,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """Load the DIM refusal direction for one model.

    Args:
        refusal_root: Root directory containing model-specific refusal
            directions. None resolves to artifacts/DIM (via
            `cast.paths.dim_direction_dir`), which is where the download script
            puts them.
        subdir: Model id, e.g. "qwen2.5-7b-instruct"
        device: Device to load the direction on
        dtype: Data type to cast the direction to

    Returns:
        Unit-normalized refusal direction tensor of shape [hidden_dim]

    Raises:
        FileNotFoundError: If direction.pt doesn't exist in the specified path
    """
    if refusal_root is None:
        direction_path = str(dim_direction_dir(subdir) / "direction.pt")
    else:
        direction_path = os.path.join(refusal_root, subdir, "direction.pt")
    if not os.path.exists(direction_path):
        raise FileNotFoundError(f"Refusal direction not found: {direction_path}")
    
    direction = torch.load(direction_path, map_location=device)
    # Cast to match vector dtype for stable projection
    direction = direction.to(device=device, dtype=dtype)
    # Normalize with numerical stability
    norm = direction.norm(dim=-1, keepdim=True)
    return direction / norm


def project_out_direction(
    vector: torch.Tensor,
    unit_direction: torch.Tensor
) -> torch.Tensor:
    """Project out the component of vector along unit_direction.

    Performs orthogonal projection: v' = v - (v · d) * d, where d is the
    unit direction.

    Args:
        vector: Input vector(s) of shape [..., hidden_dim]
        unit_direction: Unit-normalized direction of shape [hidden_dim]

    Returns:
        Projected vector(s) with same shape as input vector

    Note:
        Both tensors must be on the same device and dtype.
        unit_direction should be unit-normalized for best results.
    """
    if unit_direction.ndim != 1:
        raise ValueError(
            f"Expected a single direction of shape [hidden_dim], got "
            f"{tuple(unit_direction.shape)}"
        )

    projection_coeff = (vector * unit_direction).sum(dim=-1, keepdim=True)
    return vector - projection_coeff * unit_direction


# ============================================================================
# Ablation Functions
# ============================================================================

def process_vector_file_direction_ablation(
    src_path: str,
    dst_path: str,
    refusal_root: Optional[str],
    device: torch.device,
    overwrite: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Perform direction ablation on a single vector file.
    
    Loads the input vector, infers the appropriate refusal direction from the
    filename, projects out that direction, and saves the ablated vector. The
    ablated vector is re-normalized to preserve the original vector's magnitude.
    
    Args:
        src_path: Path to input vector file
        dst_path: Path to save ablated vector
        refusal_root: Root directory containing model-specific refusal
            directions; None uses artifacts/DIM
        device: Device for computation
        overwrite: If False, skip if dst_path already exists
    
    Returns:
        Tuple of (changed, error_message):
        - changed: True if file was processed, False if skipped
        - error_message: Error string if processing failed, None otherwise
    """
    if os.path.exists(dst_path) and not overwrite:
        return False, None
    
    # Infer which refusal direction to use
    subdir = infer_refusal_subdir_from_filename(src_path)
    if subdir is None:
        return False, f"Unable to infer refusal direction subdir from filename"
    
    # Load and flatten input vector
    vector = torch.load(src_path, map_location=device)
    if vector.ndim > 1:
        vector = vector.reshape(-1)
    
    # Store original magnitude to preserve it after ablation
    original_norm = vector.norm()
    
    # Load refusal direction
    try:
        unit_dir = load_refusal_direction(
            refusal_root, subdir, device=device, dtype=vector.dtype
        )
    except FileNotFoundError as e:
        return False, str(e)
    
    # Validate dimension compatibility
    if vector.shape[0] != unit_dir.shape[0]:
        return False, (
            f"Dimension mismatch: input vector has {vector.shape[0]} dims, "
            f"refusal direction has {unit_dir.shape[0]} dims"
        )
    
    # Perform direction ablation by projecting out the refusal direction
    ablated = project_out_direction(vector, unit_dir)
    
    # Re-normalize to preserve original magnitude
    ablated_norm = ablated.norm()
    ablated = ablated * (original_norm / ablated_norm)
    
    # Save ablated vector
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(ablated.cpu(), dst_path)
    return True, None


# ============================================================================
# Filename and Model Filtering Utilities
# ============================================================================

def modify_filename_for_ablation(
    filename: str,
    ablation_type: str,
) -> str:
    """Modify filename to record the ablation it went through.

    Appends the ablation type to the base filename before the extension; this is
    the "..._direction.pt" name run_steering.py --use_ablated_vectors expects.

    Args:
        filename: Original filename (e.g., "vec_layer_0_qwen_qwen2.5_7b_instruct.pt")
        ablation_type: Type of ablation ("direction")

    Returns:
        Modified filename with the ablation type appended
        Example: "vec_layer_0_qwen_qwen2.5_7b_instruct_direction.pt"

    Example:
        >>> modify_filename_for_ablation("vec.pt", "direction")
        'vec_direction.pt'
    """
    if not filename.endswith(SUPPORTED_FILE_EXTENSION):
        return filename

    base_name = filename[:-len(SUPPORTED_FILE_EXTENSION)]

    return base_name + "_" + ablation_type + SUPPORTED_FILE_EXTENSION


def should_process_model(filename: str, model_filter: Optional[str]) -> bool:
    """Check if a file should be processed based on model filter.
    
    Supports flexible matching:
    - "all" or None: Process all files
    - Model pattern: Match by model size (e.g., "14b", "7b") or substring
    
    Args:
        filename: Filename to check
            Example: "vec_layer_9_qwen_qwen2.5_14b_instruct.pt"
        model_filter: Model pattern to match
            Examples: "qwen2.5-14b-instruct", "qwen2.5-7b", "all", or None
    
    Returns:
        True if file should be processed, False otherwise
    
    Example:
        >>> should_process_model("vec_qwen2.5_14b.pt", "qwen2.5-14b")
        True
        >>> should_process_model("vec_qwen2.5_7b.pt", "qwen2.5-14b")
        False
    """
    if model_filter is None or model_filter.lower() == "all":
        return True
    
    # Normalize for comparison
    lower_name = filename.lower()
    lower_filter = model_filter.lower()
    
    # Normalize separators for flexible matching
    normalized_name = lower_name.replace("-", "_").replace(".", "_")
    normalized_filter = lower_filter.replace("-", "_").replace(".", "_")
    
    # Extract model size indicators (e.g., "14b", "7b", "8b")
    filter_parts = normalized_filter.split("_")
    size_parts: List[str] = []
    for part in filter_parts:
        if part.endswith("b"):
            size_str = part[:-1]  # Remove "b"
            try:
                float(size_str)  # Validate it's a number
                size_parts.append(part)
            except ValueError:
                pass
    
    # Match by size if available, otherwise use substring matching
    if size_parts:
        return any(size_part in normalized_name for size_part in size_parts)
    else:
        return normalized_filter in normalized_name


# ============================================================================
# Main Processing Function
# ============================================================================

def run(
    normalized_dir: str,
    output_dir: str,
    device_str: str = "cpu",
    overwrite: bool = False,
    model_filter: Optional[str] = None,
    ablation_type: str = "direction",
    refusal_root: Optional[str] = None,
) -> None:
    """Run ablation on all matching vector files.

    Processes vector files organized by behavior, applying the specified
    ablation type and filtering by model if requested.

    Args:
        normalized_dir: Directory containing per-behavior normalized vectors
        output_dir: Directory to write ablated vectors (mirrors input structure)
        device_str: Torch device string (e.g., "cpu", "cuda")
        overwrite: If True, overwrite existing output files
        model_filter: Model pattern to filter files (None or "all" for all models)
        ablation_type: Type of ablation ("direction")
        refusal_root: Root directory for refusal directions; None uses artifacts/DIM

    Raises:
        ValueError: If ablation_type is invalid
    """
    device = torch.device(device_str)

    require(normalized_dir, "normalized CAA steering vectors")

    # Validate ablation type. Direction ablation takes no direction path: it
    # resolves artifacts/DIM/<model_id>/direction.pt per file.
    if ablation_type not in ABLATION_TYPES:
        raise ValueError(
            f"Invalid ablation_type: {ablation_type}. "
            f"Must be one of {ABLATION_TYPES}"
        )

    # Print configuration
    print("=" * 70)
    print("Vector Ablation Configuration")
    print("=" * 70)
    print(f"Ablation type: {ablation_type}")
    print(f"Refusal direction root: {refusal_root or dim_direction_dir('<model_id>').parent}")
    if model_filter is None or model_filter.lower() == "all":
        print("Model filter: all (processing all models)")
        model_filter = "all"
    else:
        print(f"Model filter: {model_filter}")

    print(f"Input directory: {normalized_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device_str}")
    print("=" * 70)
    print()
    
    # Process files
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for behavior in sorted(os.listdir(normalized_dir)):
        behavior_src_dir = os.path.join(normalized_dir, behavior)
        if not os.path.isdir(behavior_src_dir):
            continue
        
        behavior_dst_dir = os.path.join(output_dir, behavior)
        os.makedirs(behavior_dst_dir, exist_ok=True)
        
        for fname in sorted(os.listdir(behavior_src_dir)):
            if not fname.endswith(SUPPORTED_FILE_EXTENSION):
                continue
            
            # Apply model filter
            if not should_process_model(fname, model_filter):
                skipped_count += 1
                continue
            
            src_path = os.path.join(behavior_src_dir, fname)
            
            # Modify filename to record the ablation
            modified_fname = modify_filename_for_ablation(fname, ablation_type)
            dst_path = os.path.join(behavior_dst_dir, modified_fname)

            changed, err = process_vector_file_direction_ablation(
                src_path=src_path,
                dst_path=dst_path,
                refusal_root=refusal_root,
                device=device,
                overwrite=overwrite,
            )

            # Handle results
            if err is not None:
                print(f"[ERROR] {fname}: {err}")
                error_count += 1
            elif changed:
                print(f"[WRITE] {modified_fname}")
                processed_count += 1
            else:
                print(f"[SKIP]  {modified_fname} (already exists)")
    
    # Print summary
    print()
    print("=" * 70)
    print("Processing Summary")
    print("=" * 70)
    print(f"Processed: {processed_count} files")
    if error_count > 0:
        print(f"Errors:    {error_count} files")
    if model_filter and model_filter.lower() != "all":
        print(f"Skipped:   {skipped_count} files (model filter: {model_filter})")
    print("=" * 70)


# ============================================================================
# Command-Line Interface
# ============================================================================

def main() -> None:
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the paper's 'Refusal Direction Ablation' baseline: project "
            "the fixed DIM refusal direction out of the normalized CAA steering "
            "vectors and rescale to the original norm. Output filenames carry "
            "the ablation type."
        ),
        epilog=(
            "Reproducing the Table 1 baseline for all three paper models:\n"
            "  python ablate_refusal_direction.py\n"
            "\n"
            "Directions are read from artifacts/DIM/<model_id>/direction.pt and the\n"
            "recognised model ids are llama-3.1-8b-instruct, qwen2.5-7b-instruct and\n"
            "qwen2.5-14b-instruct."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--normalized_dir",
        type=str,
        default=str(CAA_VECTOR_DIR),
        help="Directory containing per-behavior normalized CAA vectors (default: artifacts/caa_vectors)",
    )
    parser.add_argument(
        "--refusal_dir",
        type=str,
        required=False,
        default=None,
        help=(
            "Root directory holding <model_id>/direction.pt for each model "
            "(default: artifacts/DIM)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REFUSAL_ABLATED_VECTOR_DIR),
        help="Directory to write ablated vectors, mirroring the input structure (default: artifacts/refusal_ablated_vectors)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for computation (e.g., 'cpu', 'cuda', 'cuda:0')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=(
            "Model filter pattern. Use 'all' to process all models, or specify a pattern "
            "(e.g., 'qwen2.5-7b', 'llama-3.1-8b'). Pattern matching is case-insensitive and "
            "supports flexible separators."
        ),
    )
    parser.add_argument(
        "--ablation_type",
        type=str,
        default="direction",
        choices=ABLATION_TYPES,
        help=(
            "Type of ablation: 'direction' projects out the single DIM refusal "
            "direction -- the Table 1 baseline, and the only ablation the paper "
            "reports. Default: 'direction'"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files",
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run ablation
    run(
        normalized_dir=args.normalized_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        overwrite=args.overwrite,
        model_filter=args.model if args.model.lower() != "all" else None,
        ablation_type=args.ablation_type,
        refusal_root=args.refusal_dir,
    )


if __name__ == "__main__":
    main()
