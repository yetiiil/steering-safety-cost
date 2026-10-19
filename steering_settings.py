"""Configuration object shared by the steering evaluation scripts.

`SteeringSettings` carries the (behaviour, model, vector) identity of one run and
turns it into the filename suffix under which that run's results are stored.
"""

from typing import Optional, Literal
import os
from dataclasses import dataclass

from cast.paths import BEHAVIOR_DATA_DIR

# The three behaviours carrying the main results (Tables 1-14).  Also the
# default sweep of run_open_ended.py, which imports this list.
MAIN_BEHAVIORS = [
    "power-seeking-inclination",
    "corrigible-more-HHH",
    "self-awareness-good-text-model",
]

# The three further behaviours reported in Appendix B.7 (Tables 15-16).
ADDITIONAL_BEHAVIORS = [
    "sycophancy",
    "myopic-reward",
    "length_conciseness",
]

# The behaviour surface of this release: every behaviour the paper reports, and
# no others.  This is the single source of truth -- the evaluation scripts
# derive their `--behaviors` defaults and validation from it.
PAPER_BEHAVIORS = MAIN_BEHAVIORS + ADDITIONAL_BEHAVIORS


def all_behaviors() -> list:
    """Paper behaviours whose data is present under `data/behaviors/`.

    Resolved on every call rather than at import time, so that a `data/`
    directory populated after the module was first imported is still picked up.
    Falls back to the full list when the directory has not been populated yet,
    so that `--help` and the `--behaviors` defaults work on a fresh checkout.
    """
    if BEHAVIOR_DATA_DIR.is_dir():
        found = sorted(
            child.name
            for child in BEHAVIOR_DATA_DIR.iterdir()
            if child.is_dir() and child.name in PAPER_BEHAVIORS
        )
        if found:
            return found
    return list(PAPER_BEHAVIORS)


@dataclass
class SteeringSettings:
    behavior: str = "sycophancy"
    type: Literal["open_ended", "ab"] = "ab"
    system_prompt: Optional[Literal["pos", "neg"]] = None
    override_vector: Optional[int] = None
    override_vector_model: Optional[str] = None
    model_size: str = "7b"
    override_model_weights_path: Optional[str] = None
    attack_type: Optional[str] = None
    model_family: Optional[str] = None
    custom_vector_path: Optional[str] = None
    cone_dim: Optional[int] = None

    def __post_init__(self):
        # Allow custom behaviors when using custom_vector_path
        if self.custom_vector_path is None:
            known = all_behaviors()
            assert self.behavior in known, (
                f"Invalid behavior {self.behavior}; available behaviours are {known}. "
                "Populate data/behaviors/ (see the README section 'Obtaining data and "
                "artifacts') or pass a custom_vector_path."
            )
        
    def make_result_save_suffix(
        self,
        layer: Optional[int] = None,
        multiplier: Optional[float] = None,
        model_family: Optional[str] = None,
    ):
        # A custom vector is identified in the filename by the directory it sits
        # in, which is how the CAST runs keep their variants apart.
        custom_vec_name = None
        if self.custom_vector_path is not None:
            custom_vec_name = os.path.basename(os.path.dirname(self.custom_vector_path))

        combined_model = None
        fam = model_family or self.model_family
        size = self.model_size
        if fam and size:
            combined_model = f"{fam}_{size}"
        
        elements = {
            "layer": layer,
            "multiplier": multiplier,
            "behavior": self.behavior,
            "system_prompt": self.system_prompt,
            "override_vector": self.override_vector,
            "override_vector_model": self.override_vector_model,
            "model": combined_model,
            "override_model_weights_path": self.override_model_weights_path,
            "attack_type": self.attack_type,
            "custom_vector": custom_vec_name,
        }
        return "_".join([f"{k}={str(v).replace('/', '-')}" for k, v in elements.items() if v is not None])
