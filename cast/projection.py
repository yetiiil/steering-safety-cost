"""Subspace removal and renormalisation — Eq. (1) of the paper.

    ṽ  = v − Σ_i b_i b_iᵀ v
    v* = ṽ · ‖v‖ / ‖ṽ‖

`get_safe_vector` is the single implementation of that operation.  It is used
both inside the optimisation loop (`train_cast.py`, where `direction_basis` is
the current value of the learned parameter) and offline (`sanitize_vector.py`,
where `direction_basis` is a trained basis loaded from disk), so that a released
`sanitized_*.pt` can be regenerated within floating-point tolerance from `final_vector_*.pt` plus the
original CAA vector.
"""

import einops
import torch


def projection_einops(activation, direction):
    """Project `activation` onto the (unit) `direction`."""
    proj = (
        einops.einsum(
            activation, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
        * direction
    )
    return proj


def get_safe_vector(
    steering_vector: torch.Tensor,
    direction_basis: torch.Tensor,
    renorm: bool = True,
) -> torch.Tensor:
    """Remove the component of `steering_vector` lying in span(`direction_basis`).

    For a single direction d (shape [dim]) this is

        v_perp = v − proj_d(v)

    For k directions R (shape [k, dim] or [dim, k]) an orthonormal basis Q for
    span(R) is obtained by QR and

        v_perp = v − Q Qᵀ v

    Args:
        steering_vector: the vector to sanitize, shape [dim].
        direction_basis: [dim], [k, dim] or [dim, k]; also accepts a list of
            tensors, which is how the optimiser stores its basis.
        renorm: if True (the paper's default), rescale the result back to the
            norm of the input.  Steering strength is set by the multiplier α, so
            preserving ‖v‖ keeps the sanitized vector comparable to the original.

    Returns:
        The sanitized vector, same shape as `steering_vector`.
    """
    v0 = steering_vector.to(steering_vector.device, steering_vector.dtype)
    v0_norm = v0.norm()
    dim = v0.shape[0]

    basis = direction_basis
    if isinstance(basis, list):
        basis = torch.stack(basis, dim=0)

    if len(basis.shape) == 1:
        # Rank-1: the configuration used for every headline result.
        d = basis.to(device=v0.device, dtype=torch.float32)
        d = d / (d.norm() + 1e-8)
        proj = projection_einops(v0, d)
        v_perp = v0 - proj
    else:
        # Rank-k. Accept [k, dim] or [dim, k] and normalise to R: [dim, k].
        # QR is not implemented for BFloat16, hence the cast to float32.
        B = basis.to(device=v0.device, dtype=torch.float32)
        if B.shape[-1] == dim:
            R = B.transpose(0, 1)
        elif B.shape[0] == dim:
            R = B
        else:
            raise ValueError(
                f"direction_basis has incompatible shape {tuple(B.shape)} for dim={dim}"
            )

        Q, _ = torch.linalg.qr(R, mode="reduced")  # Q: [dim, k], orthonormal columns
        proj = Q @ (Q.transpose(0, 1) @ v0.to(torch.float32))  # [dim]
        v_perp = v0 - proj.to(v0.dtype)

    if not renorm:
        return v_perp

    v_perp_norm = v_perp.norm()
    scale = v0_norm / (v_perp_norm + 1e-8)
    return v_perp * scale
