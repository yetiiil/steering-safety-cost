"""
Generate per-model GCG attack suffixes for the jailbreak robustness evaluation.

GCG is the only attack in the paper that has to be *optimised against each
model*, so it is the only one this script produces, and the only --attack_method
it accepts. None of the others needs an optimiser:

- AutoDAN and PAIR are model-independent prompt sets, consumed as static files
  from data/attacks/ (jailbreakbench_autodan.csv, jailbreakbench_pair_<model>.json)
- prefix injection, refusal suppression and AIM are static templates that
  run_steering.py wraps around each goal at evaluation time
  (--attack_method prefix / refusal_suppression / aim)

Two harmful-behaviour sources are supported via --dataset:
- jailbreakbench: the JBB-Behaviors harmful split (100 behaviours), used for the
  reported numbers
- harmbench: the HarmBench standard split, which run_steering.py also accepts.
  The paper's jailbreak results are JailbreakBench only

The output filename always carries the dataset tag
(data/attacks/gcg/gcg_<model>_<dataset>.csv), because that is what
run_steering.py reads for both datasets.
"""

import argparse
import nanogcg
import os
import pandas as pd
from dotenv import load_dotenv
from typing import Optional

from datasets import load_dataset
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from cast.paths import ATTACK_DATA_DIR

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

# GCG search width, per dataset. These were separate scripts before the merge
# and were run with different widths, so the default here is per-dataset rather
# than a single number: each reproduces the suffixes that dataset's released
# artifacts were built with. Override with --gcg_search_width.
GCG_SEARCH_WIDTH = {
    "jailbreakbench": 128,
    "harmbench": 64,
}
GCG_NUM_STEPS = 100


def load_model(model_name: str, use_8bit: bool = False):
    """Load model by name (e.g., 'qwen/Qwen2.5-7b-instruct').
    
    Args:
        model_name: HuggingFace model name or identifier
        use_8bit: Whether to use 8-bit quantization (currently not fully supported)
    """
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with device_map instead of .to("cuda")
    if use_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=False,  # Set to True if 8-bit quantization is needed
            device_map="cuda",
            token=HUGGINGFACE_TOKEN,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
            token=HUGGINGFACE_TOKEN,
        )
    
    return model, tokenizer


def load_jailbreakbench_data():
    """Load JailbreakBench harmful behaviors dataset."""
    jbb = pd.read_csv("hf://datasets/JailbreakBench/JBB-Behaviors/" + "data/harmful-behaviors.csv")
    return jbb


def load_harmbench_data():
    """Load the HarmBench standard split in the same (Goal, Target) shape as JBB.

    HarmBench ships no target string, so one is synthesised the way the GCG
    literature does: "Sure, here is " followed by the behaviour with its leading
    word (the imperative verb) stripped.
    """
    ds = load_dataset("walledai/HarmBench", name="standard", split="train")

    df = ds.to_pandas()
    df["target"] = "Sure, here is " + df["prompt"].str.replace(
        r"^\S+\s*", "", regex=True
    )

    df = df.rename(columns={"prompt": "Goal", "target": "Target"})

    print(f"Loaded {len(df)} templates from HarmBench")
    return df


DATASET_LOADERS = {
    "jailbreakbench": load_jailbreakbench_data,
    "harmbench": load_harmbench_data,
}


def generate_gcg_attack(
    model,
    tokenizer,
    message: str,
    target: str,
    search_width: int = GCG_SEARCH_WIDTH["jailbreakbench"],
    num_steps: int = GCG_NUM_STEPS,
    **kwargs,
) -> str:
    """Generate a single GCG attack string.

    Args:
        model: The language model
        tokenizer: The tokenizer
        message: The goal/prompt
        target: The target response
        search_width: GCG candidate batch size per step
        num_steps: Maximum GCG optimisation steps (early_stop can end it sooner)
        **kwargs: Additional arguments

    Returns:
        Attack string
    """
    config = GCGConfig(
        verbosity="WARNING",
        search_width=search_width,
        num_steps=num_steps,
        early_stop=True,
    )
    result = nanogcg.run(model, tokenizer, message, target, config)
    return result.best_string


# Attack method registry. GCG is the only entry because it is the only attack
# that is optimised per model; everything else in the paper is either a static
# template applied by run_steering.py or a static prompt file (see the module
# docstring). Registering an attack here means this script can generate it.
ATTACK_METHODS = {
    "gcg": generate_gcg_attack,
}


def generate_attacks(
    model,
    tokenizer,
    jbb,
    attack_method: str,
    model_name: Optional[str] = None,
    output_path: Optional[str] = None,
    dataset: str = "jailbreakbench",
    **kwargs
):
    """Generate attack strings for all behaviors in the dataset.

    Args:
        model: The language model
        tokenizer: The tokenizer
        jbb: Behaviour dataframe with 'Goal' and 'Target' columns
        attack_method: Attack method to use (e.g., "gcg")
        model_name: Model name for column naming (e.g., "qwen/qwen2.5-7b-instruct")
        output_path: Output CSV path (defaults to
            data/attacks/{attack_method}/{attack_method}_{model}_{dataset}.csv,
            the name run_steering.py reads)
        dataset: Which behaviour source the rows came from; it is part of the
            column name and filename so the two runs never overwrite each other
        **kwargs: Additional arguments passed to the attack function
    """
    if attack_method not in ATTACK_METHODS:
        raise ValueError(
            f"Unknown attack method: {attack_method}. "
            f"Available methods: {', '.join(ATTACK_METHODS.keys())}"
        )

    if model_name is None:
        model_name = "model"

    # Sanitize model name for filename and column name
    model_name_sanitized = model_name.replace("/", "_").replace("-", "_")
    # The *column* name is tagged only for HarmBench: the released artifacts use
    # <model>_gcg_string for JailbreakBench and <model>_gcg_harmbench_string for
    # HarmBench, and run_steering.py accepts either spelling.
    column_tag = "" if dataset == "jailbreakbench" else f"_{dataset}"
    column_name = f"{model_name_sanitized}_{attack_method}{column_tag}_string"

    if output_path is None:
        # The *filename* is tagged for every dataset, including the default one:
        # run_steering.load_gcg_data_from_csv reads gcg_<model>_<dataset>.csv
        # unconditionally, and the released artifacts are named that way.
        output_path = str(
            ATTACK_DATA_DIR
            / attack_method
            / f"{attack_method}_{model_name_sanitized}_{dataset}.csv"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Check if output file exists and load existing results
    start_idx = 0
    if os.path.exists(output_path):
        print(f"Found existing output file: {output_path}")
        existing_df = pd.read_csv(output_path)
        
        # Check if the attack string column exists in the existing file
        if column_name in existing_df.columns:
            # Find the first empty/null value in the attack string column
            # Empty values can be: NaN, None, empty string, or whitespace-only string
            empty_mask = (
                existing_df[column_name].isna() | 
                (existing_df[column_name].astype(str).str.strip() == '') |
                (existing_df[column_name].astype(str).str.strip() == 'nan')
            )
            
            if empty_mask.any():
                # Find the first index where the value is empty
                empty_indices = existing_df.index[empty_mask]
                start_idx = empty_indices[0]
                print(f"Resuming from row {start_idx + 1} (index {start_idx})")
                # Merge existing results into jbb, ensuring indices align
                if len(existing_df) == len(jbb):
                    jbb[column_name] = existing_df[column_name].values
                else:
                    print(f"Warning: Existing file has {len(existing_df)} rows, but dataset has {len(jbb)} rows.")
                    print("Merging by index where possible...")
                    jbb[column_name] = None
                    for idx in existing_df.index:
                        if idx < len(jbb):
                            jbb.loc[idx, column_name] = existing_df.loc[idx, column_name]
            else:
                print("All attack strings are already generated. Exiting.")
                return jbb
        else:
            print(f"Column {column_name} not found in existing file. Starting from beginning.")
            # Initialize the column
            jbb[column_name] = None
    else:
        print(f"No existing output file found. Starting from beginning.")
        # Initialize the column
        jbb[column_name] = None
    
    # Get the attack function
    attack_fn = ATTACK_METHODS[attack_method]
    
    # Process from start_idx to the end
    for i in range(start_idx, len(jbb)):
        row = jbb.iloc[i]
        message = row["Goal"]
        target = row["Target"]
        
        # Skip if already has a non-empty attack string
        if pd.notna(jbb.loc[i, column_name]) and str(jbb.loc[i, column_name]).strip() != '':
            print(f"Skipping behavior {i+1}/{len(jbb)} (already has attack string): {row.get('Behavior', 'N/A')}")
            continue
        
        print(f"Processing behavior {i+1}/{len(jbb)}: {row.get('Behavior', 'N/A')}")
        
        try:
            attack_string = attack_fn(model, tokenizer, message, target, **kwargs)
            jbb.loc[i, column_name] = attack_string
        except NotImplementedError as e:
            print(f"Error: {e}")
            jbb.loc[i, column_name] = None
        except Exception as e:
            print(f"Error processing behavior {i+1}: {e}")
            jbb.loc[i, column_name] = None

        # Save after each row to allow resuming
        jbb.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")
    return jbb


def main():
    """Main function to run attack generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Optimise per-model GCG attack suffixes for a harmful-behaviour "
            "dataset. This is the only attack in the paper that is generated "
            "here. AutoDAN and PAIR prompts are model-independent and are "
            "consumed as static files from data/attacks/ "
            "(jailbreakbench_autodan.csv, jailbreakbench_pair_<model>.json); "
            "prefix injection, refusal suppression and AIM are static templates "
            "that run_steering.py applies at evaluation time (--attack_method "
            "prefix / refusal_suppression / aim)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., 'qwen/qwen2.5-7b-instruct')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="jailbreakbench",
        choices=list(DATASET_LOADERS),
        help=(
            "Harmful-behaviour source. 'jailbreakbench' is the JBB-Behaviors "
            "harmful split, the only one the paper's jailbreak results use; "
            "'harmbench' is the HarmBench standard split, which run_steering.py "
            "also accepts. Both are downloaded from the Hub. "
            "Default: jailbreakbench"
        ),
    )
    parser.add_argument(
        "--attack_method",
        type=str,
        required=True,
        choices=list(ATTACK_METHODS.keys()),
        help=(
            f"Attack method to generate. Available: "
            f"{', '.join(ATTACK_METHODS.keys())}. The paper's other attacks are "
            "not generated: prefix, refusal_suppression and aim are static "
            "templates applied by run_steering.py, and autodan/pair are static "
            "prompt files under data/attacks/."
        ),
    )
    parser.add_argument(
        "--gcg_search_width",
        type=int,
        default=None,
        help=(
            "GCG candidates evaluated per step. Default is the value the "
            "released suffixes were produced with: "
            f"{GCG_SEARCH_WIDTH['jailbreakbench']} for --dataset jailbreakbench, "
            f"{GCG_SEARCH_WIDTH['harmbench']} for --dataset harmbench."
        ),
    )
    parser.add_argument(
        "--gcg_num_steps",
        type=int,
        default=GCG_NUM_STEPS,
        help=f"Maximum GCG optimisation steps (default: {GCG_NUM_STEPS})",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Output CSV path (defaults to "
            "data/attacks/{attack_method}/{attack_method}_{model}_{dataset}.csv, "
            "which is the name run_steering.py reads)"
        ),
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        default=False,
        help="Use 8-bit quantization (currently not fully supported)",
    )

    args = parser.parse_args()

    search_width = args.gcg_search_width
    if search_width is None:
        search_width = GCG_SEARCH_WIDTH[args.dataset]

    print(f"Loading model: {args.model_name}...")
    model, tokenizer = load_model(args.model_name, use_8bit=args.use_8bit)
    print("Model loaded successfully.")

    jbb = DATASET_LOADERS[args.dataset]()
    print(f"Loaded {len(jbb)} behaviors.")

    # Generate attacks
    print(f"Generating {args.attack_method.upper()} attack strings...")
    generate_attacks(
        model,
        tokenizer,
        jbb,
        attack_method=args.attack_method,
        model_name=args.model_name,
        output_path=args.output_path,
        dataset=args.dataset,
        search_width=search_width,
        num_steps=args.gcg_num_steps,
    )


if __name__ == "__main__":
    main()
