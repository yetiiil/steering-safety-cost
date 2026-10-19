#!/usr/bin/env python3
"""Sanitize a CAA steering vector with a trained CAST direction — Eq. (1).

This is the cheapest way to verify the paper's artifacts.  Given a released
`final_vector_*.pt` (the direction r̂ produced by `train_cast.py`) and the CAA
steering vector it was trained against, this script regenerates the released
`sanitized_*.pt` within floating-point tolerance. No model or GPU is needed: it is a
projection plus a rescale on two vectors of a few thousand floats, so it runs on
CPU in seconds.

    python sanitize_vector.py \
        --steering_vector_path artifacts/caa_vectors/myopic-reward/vec_layer_18_qwen_qwen2.5_7b_instruct.pt \
        --vectors_path artifacts/cast_vectors/qwen2.5-7b-instruct/myopic-reward/final_vector_myopic-reward.pt \
        --output_path artifacts/cast_vectors/qwen2.5-7b-instruct/myopic-reward/sanitized_myopic-reward.pt \
        --device cpu

The projection itself lives in `cast.projection.get_safe_vector`, which is the
same function the optimisation loop in `train_cast.py` calls.  Keeping exactly
one implementation is what makes the check above meaningful: it shows that the
offline sanitization is the operation CAST optimised, not a re-derivation of it.
"""

import argparse
import os

import torch

from cast.paths import require
from cast.projection import get_safe_vector


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sanitize a CAA steering vector by projecting out a trained CAST "
            "direction and rescaling to the original norm (Eq. 1)."
        ),
        epilog=(
            "Verifying a released artifact:\n"
            "  python sanitize_vector.py \\\n"
            "      --steering_vector_path artifacts/caa_vectors/<behavior>/vec_layer_<L>_<model>.pt \\\n"
            "      --vectors_path artifacts/cast_vectors/<model_id>/<behavior>/final_vector_<behavior>.pt \\\n"
            "      --output_path /tmp/sanitized_<behavior>.pt --device cpu\n"
            "and compare the result against the released sanitized_<behavior>.pt.\n"
            "This needs no model and no GPU; it takes seconds on CPU."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steering_vector_path",
        type=str,
        required=True,
        help="Path to the CAA steering vector (.pt file) to be sanitized"
    )
    parser.add_argument(
        "--vectors_path",
        type=str,
        required=True,
        help=(
            "Path to the trained CAST direction (.pt file): a released "
            "final_vector_*.pt, or vectors.pt / lowest_loss_vector.pt from a "
            "training run"
        )
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the sanitized steering vector (default: adds '_sanitized.pt' to steering_vector_path)"
    )
    parser.add_argument(
        "--use_lowest_loss",
        action="store_true",
        help="Accepted for backwards compatibility; selecting by loss is not implemented, so the last checkpoint in vectors.pt is used either way"
    )
    parser.add_argument(
        "--no_renorm",
        action="store_true",
        help="Don't renormalize the sanitized vector to match original norm"
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Model dtype used during training (paper: bfloat16). The CAA vector "
             "is rounded to this dtype before float32 projection, as in train_cast.py.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation (default: cuda if available, else cpu). 'cpu' is enough for this script."
    )

    args = parser.parse_args()

    require(args.steering_vector_path, "CAA steering vector")
    require(args.vectors_path, "trained CAST direction")

    # Load steering vector
    print(f"Loading steering vector from {args.steering_vector_path}")
    steering_vector = torch.load(args.steering_vector_path, map_location=args.device)
    if isinstance(steering_vector, dict):
        # Handle case where file contains a dict with 'vector' key
        if 'vector' in steering_vector:
            steering_vector = steering_vector['vector']
        else:
            raise ValueError(f"Dictionary loaded but no 'vector' key found. Keys: {steering_vector.keys()}")
    
    if len(steering_vector.shape) != 1:
        raise ValueError(f"Expected 1D steering vector, got shape {steering_vector.shape}")

    # train_cast.py first loads the CAA vector in the model's dtype, then
    # RefusalCone stores that rounded vector in float32 for optimisation.
    steering_vector = steering_vector.to(getattr(torch, args.dtype)).to(torch.float32)
    
    print(f"  Shape: {steering_vector.shape}")
    print(f"  Norm: {steering_vector.norm().item():.4f}")
    print(f"  Dtype: {steering_vector.dtype}")
    print(f"  Training dtype: {args.dtype}")

    # Load refusal basis vectors
    print(f"\nLoading refusal basis vectors from {args.vectors_path}")
    vectors = torch.load(args.vectors_path, map_location=args.device)
    
    # Handle different formats
    if isinstance(vectors, list):
        # vectors.pt holds one basis per optimisation step; the released
        # final_vector_*.pt is the last of them, so that is what we take.
        if args.use_lowest_loss:
            print("  Note: --use_lowest_loss is not implemented; using the last vector set. "
                  "Point --vectors_path at lowest_loss_vector.pt instead.")
        basis_vectors = vectors[-1]
    elif isinstance(vectors, torch.Tensor):
        if len(vectors.shape) == 2:
            # Shape [k, dim] - this is what we want
            basis_vectors = vectors
        elif len(vectors.shape) == 1:
            # Single vector, wrap it
            basis_vectors = vectors.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected tensor shape: {vectors.shape}")
    else:
        raise ValueError(f"Unexpected type for vectors: {type(vectors)}")
    
    print(f"  Basis vectors shape: {basis_vectors.shape}")
    print(f"  Number of basis vectors: {basis_vectors.shape[0] if len(basis_vectors.shape) > 1 else 1}")

    # Compute sanitized vector
    print("\nComputing sanitized steering vector...")
    sanitized_vector = get_safe_vector(
        steering_vector=steering_vector,
        direction_basis=basis_vectors,
        renorm=not args.no_renorm
    )

    print(f"\nResults:")
    print(f"  Original norm: {steering_vector.norm().item():.4f}")
    print(f"  Sanitized norm: {sanitized_vector.norm().item():.4f}")
    print(f"  Norm ratio: {sanitized_vector.norm().item() / steering_vector.norm().item():.4f}")

    # Determine output path
    if args.output_path is None:
        base_path = args.steering_vector_path
        if base_path.endswith('.pt'):
            output_path = base_path[:-3] + '_sanitized.pt'
        else:
            output_path = base_path + '_sanitized.pt'
    else:
        output_path = args.output_path

    # Save sanitized vector
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save(sanitized_vector.cpu(), output_path)
    print(f"\nSanitized steering vector saved to: {output_path}")


if __name__ == "__main__":
    main()
