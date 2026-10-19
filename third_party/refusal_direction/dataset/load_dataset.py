# Modified from the original: https://github.com/andyrdt/refusal_direction (Apache-2.0).
# See third_party/refusal_direction/UPSTREAM.md for the full list of changes.

import os
import json

dataset_dir_path = os.path.dirname(os.path.realpath(__file__))

# <repo>, three levels up from third_party/refusal_direction/dataset.
repo_root = os.path.abspath(os.path.join(dataset_dir_path, '..', '..', '..'))

SPLITS = ['train', 'val', 'test']
HARMTYPES = ['harmless', 'harmful']

# The harmful/harmless splits are the SALAD-Bench derived ones used for the paper,
# not the upstream `dataset/splits/` shipped with the DIM repository.
SPLIT_DATASET_FILENAME = os.path.join(repo_root, 'data', 'saladbench_splits', '{harmtype}_{split}.json')


def _require(file_path: str, what: str) -> str:
    """Return `file_path`, or raise pointing at the data instructions."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{what} not found at {file_path}.\n"
            "Run scripts/00_download_data.py from the repository root to "
            "restore the fixed paper inputs. See 'Obtaining data and artifacts'."
        )
    return file_path

PROCESSED_DATASET_NAMES = ["advbench", "tdc2023", "maliciousinstruct", "harmbench_val", "harmbench_test", "jailbreakbench", "strongreject", "alpaca", "xstest", "sorrybench"]

def load_dataset_split(harmtype: str, split: str, instructions_only: bool=False):
    assert harmtype in HARMTYPES
    assert split in SPLITS

    file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)
    _require(file_path, f"DIM {harmtype} {split} split")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset

def load_strongreject():
    from strong_reject.load_datasets import load_strongreject
    forbidden_prompt_dataset = load_strongreject()
    forbidden_prompts = forbidden_prompt_dataset["forbidden_prompt"]
    categories = forbidden_prompt_dataset["category"]
    return [{"instruction": prompt, "category": category} for prompt, category in zip(forbidden_prompts, categories)]

def load_dataset(dataset_name, instructions_only: bool=False):
    assert dataset_name in PROCESSED_DATASET_NAMES, f"Valid datasets: {PROCESSED_DATASET_NAMES}"

    if dataset_name == "strongreject":
        return load_strongreject()

    # The fixed JailbreakBench evaluation input is restored in this layout.
    file_path = os.path.join(dataset_dir_path, 'processed', f"{dataset_name}.json")
    _require(file_path, f"Evaluation dataset '{dataset_name}'")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]
 
    return dataset
