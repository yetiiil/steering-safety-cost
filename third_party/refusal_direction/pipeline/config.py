# Modified from the original: https://github.com/andyrdt/refusal_direction (Apache-2.0).
# See third_party/refusal_direction/UPSTREAM.md for the full list of changes.

import os

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    model_alias: str
    model_path: str
    n_test: int = 128
    splits: str = "saladbench"
    sample: bool = False
    filter_train: bool = True
    filter_val: bool = True
    # evaluation_datasets: Tuple[str] = ("jailbreakbench", "strongreject", "sorrybench", "xstest")
    evaluation_datasets: Tuple[str] = ("jailbreakbench",)
    evaluate_ablation: bool = True
    evaluate_actadd: bool = False
    evaluate_harmless: bool = False
    evaluate_loss: bool = False
    
    subspace_n_samples: int = 64
    max_new_tokens: int = 512
    completions_batch_size: int = 128

    jailbreak_eval_methodologies: Tuple[str] = ("substring_matching", "strongreject")
    refusal_eval_methodologies: Tuple[str] = ("substring_matching",)
    ce_loss_batch_size: int = 2
    ce_loss_n_batches: int = 2048

    def artifact_path(self) -> str:
        # Same resolution as cast.paths.dim_direction_dir, so that the direction
        # written here is the one train_cast.py later reads: SAVE_DIR/DIM_DIR when
        # SAVE_DIR is set in the environment, else <repo>/artifacts/DIM.  The repo
        # root is three levels above this file (third_party/refusal_direction/pipeline).
        repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))
        save_dir = os.getenv("SAVE_DIR") or os.path.join(repo_root, "artifacts")
        dim_dir = os.getenv("DIM_DIR", "DIM")
        return os.path.join(save_dir, dim_dir, self.model_alias)
