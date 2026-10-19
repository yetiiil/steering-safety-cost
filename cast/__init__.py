"""CAST — Constrained Ablation for Safe STeering.

Shared building blocks for the code accompanying "Could Inference-Time
Interventions Preserve Alignment? Safety Cost of Steering Vectors Is Separable
and Reducible" (Li & Kasneci, COLM 2026).

The top-level scripts in the repository root are the entry points; this package
holds the pieces more than one of them needs, so that the sanitization step in
`sanitize_vector.py` is provably the same operation as the one inside the
training loop.
"""

from cast.projection import get_safe_vector, projection_einops

__all__ = ["get_safe_vector", "projection_einops"]
