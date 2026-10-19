#!/usr/bin/env python3
"""
Script to generate raw CAA steering vectors for all behaviors on the paper's
three models: Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct and Llama-3.1-8B-Instruct.

This script generates steering vectors by extracting activations from positive and negative
examples in behavior datasets, following the approach used in the CAA paper.

The vectors written here are *raw*: their norms differ from behavior to behavior,
so a given multiplier means something different for each. Run normalize_vectors.py
afterwards to put every behavior on the same norm; everything downstream
(train_cast.py, run_steering.py, ...) reads the normalized vectors.

Example usage:
# Generate vectors for all behaviors on llama-3.1-8b-instruct
python generate_vectors.py --model_family llama --model_size 8b --behaviors all

# Generate vectors for specific behaviors on qwen-2.5-7b-instruct
python generate_vectors.py --model_family qwen --model_size 7b --behaviors myopic-reward sycophancy

# Generate vectors for selected layers of a specific behavior
python generate_vectors.py --model_family llama --model_size 8b --behaviors myopic-reward --layers $(seq 0 31)
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports
from steering_vectors import train_steering_vector, SteeringVector
from tokenizer import create_chat_prompt

from cast.paths import ARTIFACTS_DIR, BEHAVIOR_DATA_DIR, behavior_dir, require

# Puts HF_TOKEN from .env into the environment, where huggingface_hub picks it
# up; the gated Llama checkpoints cannot be downloaded without it.
load_dotenv()

# Raw, un-normalized output. normalize_vectors.py reads this tree and writes
# cast.paths.CAA_VECTOR_DIR (artifacts/caa_vectors), which is what the rest of
# the repository consumes.
RAW_CAA_VECTOR_DIR = ARTIFACTS_DIR / "caa_vectors_raw"


def get_ab_data_path(behavior: str) -> str:
    """Return path to the contrastive A/B dataset for a behavior."""
    return str(behavior_dir(behavior) / "generate_dataset.json")


def list_all_behaviors() -> List[str]:
    """List behaviors under data/behaviors that have a generate_dataset.json."""
    root = BEHAVIOR_DATA_DIR
    if not root.exists():
        return []
    behaviors = []
    for child in root.iterdir():
        if child.is_dir() and (child / "generate_dataset.json").exists():
            behaviors.append(child.name)
    return sorted(behaviors)


def get_vector_dir(behavior: str) -> str:
    """Get the directory path for saving raw vectors."""
    return str(RAW_CAA_VECTOR_DIR / behavior)


def get_vector_path(behavior: str, layer: int, model_name_path: str) -> str:
    """Get the file path for saving a specific vector."""
    # Clean model name for filename
    model_name_clean = model_name_path.replace("/", "_").replace("-", "_")
    filename = f"vec_layer_{layer}_{model_name_clean}.pt"
    return str(Path(get_vector_dir(behavior)) / filename)


class VectorGenerator:
    """Universal steering vector generator for all behaviors and model families."""

    def __init__(self, model_family: str, model_size: str, use_chat: bool = True):
        self.model_family = model_family.lower()
        self.model_size = model_size
        self.use_chat = use_chat
        self.model = None
        self.tokenizer = None
        self.model_name_path = None

    def setup_model_and_tokenizer(self):
        """Load the model and tokenizer based on family and size."""
        # Normalize model size to lowercase for model names
        model_size_lower = self.model_size.lower()

        if self.model_family == "qwen":
            if model_size_lower in ("7b", "14b"):
                self.model_name_path = f"qwen/qwen2.5-{model_size_lower}-instruct"
            else:
                raise ValueError(f"Unsupported model size: {model_size_lower}")
        elif self.model_family == "llama":
            if model_size_lower == "8b":
                self.model_name_path = f"meta-llama/llama-3.1-8b-instruct"
            else:
                raise ValueError(f"Unsupported model size: {model_size_lower}")
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")

        print(f"Loading model: {self.model_name_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_path)

        # Full precision: the released vectors were extracted without
        # quantization, and 8-bit activations shift the contrastive difference.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_path,
            load_in_8bit=False,
            device_map="auto",
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            if (
                hasattr(self.tokenizer, "eos_token")
                and self.tokenizer.eos_token is not None
            ):
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                # Add a pad token
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.model.resize_token_embeddings(len(self.tokenizer))

    def load_behavior_dataset(self, behavior: str) -> List[Tuple[str, str]]:
        """
        Load a behavior dataset and convert it to positive/negative pairs.

        Args:
            behavior: The behavior name (e.g., "myopic-reward", "sycophancy")

        Returns:
            List of (positive_prompt, negative_prompt) tuples
        """
        data_path = require(
            get_ab_data_path(behavior),
            f"contrastive dataset for behavior '{behavior}'",
        )

        with open(data_path, "r") as f:
            data = json.load(f)

        pairs = []
        for item in data:
            question = item["question"]
            pos_answer = item["answer_matching_behavior"]
            neg_answer = item["answer_not_matching_behavior"]

            # Use universal tokenizer for all model families
            pos_prompt = create_chat_prompt(
                tokenizer=self.tokenizer,
                question=question,
                answer=pos_answer,
                system_prompt=None,
                use_chat_format=self.use_chat,
            )
            neg_prompt = create_chat_prompt(
                tokenizer=self.tokenizer,
                question=question,
                answer=neg_answer,
                system_prompt=None,
                use_chat_format=self.use_chat,
            )

            pairs.append((pos_prompt, neg_prompt))

        return pairs

    def create_directories(self, behavior: str):
        """Create necessary directories for saving steering vectors."""
        vector_dir = get_vector_dir(behavior)
        os.makedirs(vector_dir, exist_ok=True)

    def generate_vectors_for_behavior(
        self,
        behavior: str,
        layers: List[int],
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> Optional[SteeringVector]:
        """
        Generate steering vectors for a specific behavior.

        Args:
            behavior: The behavior name
            layers: List of layer indices to generate vectors for
            batch_size: Batch size for processing
            show_progress: Whether to show progress bars

        Returns:
            SteeringVector object or None if generation failed
        """
        try:
            print(f"\nGenerating vectors for behavior: {behavior}")

            # Load dataset
            dataset = self.load_behavior_dataset(behavior)
            print(f"Loaded {len(dataset)} training samples")

            # Create directories
            self.create_directories(behavior)

            # Generate steering vectors
            print("Generating steering vectors...")
            steering_vector = train_steering_vector(
                model=self.model,
                tokenizer=self.tokenizer,
                training_samples=dataset,
                layers=layers,
                layer_type="decoder_block",
                move_to_cpu=False,
                read_token_index=-2,  # Second-to-last token, following CAA approach
                show_progress=show_progress,
                batch_size=batch_size,
                tqdm_desc=f"Training {behavior} steering vector",
            )

            # Save vectors for each layer
            for layer in layers:
                if layer in steering_vector.layer_activations:
                    vector_path = get_vector_path(behavior, layer, self.model_name_path)
                    torch.save(steering_vector.layer_activations[layer], vector_path)
                    print(f"Saved steering vector for layer {layer} to {vector_path}")
                else:
                    print(f"Warning: No vector generated for layer {layer}")

            print(f"Successfully generated vectors for {behavior}")
            return steering_vector

        except Exception as e:
            print(f"Error generating vectors for {behavior}: {e}")
            return None

    def generate_vectors(
        self,
        behaviors: List[str],
        layers: List[int],
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> Dict[str, Optional[SteeringVector]]:
        """
        Generate steering vectors for multiple behaviors.

        Args:
            behaviors: List of behavior names
            layers: List of layer indices to generate vectors for
            batch_size: Batch size for processing
            show_progress: Whether to show progress bars

        Returns:
            Dictionary mapping behavior names to SteeringVector objects
        """
        # Setup model and tokenizer once
        self.setup_model_and_tokenizer()

        results = {}
        for behavior in behaviors:
            results[behavior] = self.generate_vectors_for_behavior(
                behavior=behavior,
                layers=layers,
                batch_size=batch_size,
                show_progress=show_progress,
            )

        return results


def get_default_layers(model_family: str, model_size: str) -> List[int]:
    """Get default layer range based on model family and size."""
    if model_family == "qwen":
        if model_size == "7b":
            return list(range(28))  # qwen2.5-7B: 28 layers
        elif model_size == "14b":
            return list(range(48))  # qwen2.5-14B: 48 layers
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
    elif model_family == "llama":
        if model_size == "8b":
            return list(range(32))  # llama-3.1-8B: 32 layers
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
    else:
        raise ValueError(f"Unsupported model family: {model_family}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate steering vectors for all behaviors and model families"
    )

    # Model configuration
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["llama", "qwen"],
        default="llama",
        help="Model family to use (default: llama)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["7b", "8b", "14b"],
        default="8b",
        help="Size of the model to use: qwen 7b or 14b, llama 8b (default: 8b)",
    )

    # Behavior configuration
    parser.add_argument(
        "--behaviors",
        nargs="+",
        type=str,
        default=["all"],
        help="List of behaviors to generate vectors for, or 'all' for all behaviors",
    )

    # Layer configuration
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="List of layer indices to generate vectors for (default: all layers for model)",
    )

    # Processing configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        default=False,
        help="Disable progress bars",
    )

    args = parser.parse_args()

    # Resolved here rather than at import time, so that --help works on a
    # checkout whose data/ directory has not been populated yet.
    all_behaviors = list_all_behaviors()
    if not all_behaviors:
        raise FileNotFoundError(
            f"No behaviors found under {BEHAVIOR_DATA_DIR}.\n"
            "This repository ships code only. See the README section "
            "'Obtaining data and artifacts' and run scripts/00_download_data.py."
        )

    # Handle behaviors
    if "all" in args.behaviors:
        behaviors = all_behaviors
    else:
        behaviors = args.behaviors
        # Validate behaviors
        invalid_behaviors = [b for b in behaviors if b not in all_behaviors]
        if invalid_behaviors:
            print(f"Error: Invalid behaviors: {invalid_behaviors}")
            print(f"Valid behaviors: {all_behaviors}")
            return

    # Handle layers
    if args.layers is None:
        layers = get_default_layers(args.model_family, args.model_size)
    else:
        layers = args.layers

    print(f"Configuration:")
    print(f"  Model family: {args.model_family}")
    print(f"  Model size: {args.model_size}")
    print(f"  Behaviors: {behaviors}")
    print(f"  Layers: {layers}")
    print(f"  Batch size: {args.batch_size}")

    # Generate vectors
    generator = VectorGenerator(
        model_family=args.model_family,
        model_size=args.model_size,
    )

    results = generator.generate_vectors(
        behaviors=behaviors,
        layers=layers,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)

    successful = 0
    failed = 0

    for behavior, steering_vector in results.items():
        if steering_vector is not None:
            print(f"[OK]   {behavior}: Success")
            successful += 1
        else:
            print(f"[FAIL] {behavior}: Failed")
            failed += 1

    print(f"\nTotal: {successful} successful, {failed} failed")

    if successful > 0:
        print(f"\nRaw vectors saved to: {RAW_CAA_VECTOR_DIR}")
        print("Next: python normalize_vectors.py --model_family ... --model_size ... --n_layers ...")


if __name__ == "__main__":
    main()
