import torch
from typing import Optional
from torch_geometric.utils import scatter

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """Broadcast `src` to the shape of `other` starting at dimension `dim`."""
    if dim < 0:
        dim = other.dim() + dim
    for _ in range(dim):
        src = src.unsqueeze(0)
    while src.dim() < other.dim():
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Scatter-aware softmax: compute softmax scores grouped by `index` along `dim`."""

    if not torch.is_floating_point(src):
        raise ValueError("`scatter_softmax` requires floating-point input tensors.")

    max_value_per_index = scatter(src, index, dim=dim, dim_size=dim_size, reduce="max")

    expanded_index = broadcast(index, src, dim)

    max_per_src_element = max_value_per_index.gather(dim, expanded_index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp()

    sum_per_index = scatter(
        recentered_scores_exp, index, dim=dim, dim_size=dim_size, reduce="sum"
    )
    normalizing_constants = sum_per_index.gather(dim, expanded_index)

    return recentered_scores_exp / normalizing_constants



def calc_weights(data) -> torch.Tensor:
    """
    Compute per-atom radial weights relative to each molecule's geometric center.

    The weight for each atom is the Euclidean distance from the atom to the
    geometric center (mean position) of its molecule, normalized by the maximum
    distance within that molecule. Normalization ensures weights are in [0, 1].
    If an atom lays on the geometric center, it's weights is 0.

    Parameters
    ----------
    data : AtomicData
        AtomicData containing `pos` (tensor of shape [N, 3]) and `batch`
        (tensor of length N) mapping atoms to molecule indices.

    Returns
    -------
    torch.Tensor
        Tensor of shape [N] with a normalized radial weight for each atom.
    """
    virt_pos = scatter(data.pos, data.atom_batch, reduce="mean", dim=0)

    weights = (data.pos - virt_pos[data.atom_batch]).norm(p=2, dim=1)
    norm = scatter(weights, data.atom_batch, reduce="max")[data.atom_batch]
    norm = torch.where(norm == 0.0, torch.ones_like(norm), norm)
    weights = weights / norm

    return weights


def calc_weights_pbc(data) -> torch.Tensor:
    """
    Compute per-atom radial weights with periodic boundary conditions (PBC).

    This function accounts for periodic images when computing distances to the
    geometric center. Positions are transformed to fractional coordinates using
    the inverse cell, the mean (geometric center) is computed in fractional
    space, and distances are corrected using minimal image convention. Final
    distances are converted back to Cartesian space and normalized by the
    per-molecule maximum distance to yield values in [0, 1].

    Requirements on `data`
    ---------------------
    - data.cell indicating cell vectors.
    - data.pbc indicating periodicity along each axis.
    - data.pos and data.batch as in calc_weights.

    Parameters
    ----------
    data : AtomicData
        AtomicData providing `pos`, `batch`, `cell` and `pbc`.

    Returns
    -------
    torch.Tensor
        Tensor of shape [N] with normalized radial weights for each atom that
        respect periodic boundary conditions.
    """
    cell = data.cell.view((-1, 3, 3)).transpose(2, 1)
    pbc = torch.ones((cell.shape[0], 3, 1), device=cell.device)

    reciprocal_cell = torch.linalg.inv(cell)
    scaled_pos = reciprocal_cell[data.atom_batch] @ data.pos.unsqueeze(-1)
    scaled_virt_pos = scatter(scaled_pos, data.atom_batch, reduce="mean", dim=0)
    diff = scaled_pos - scaled_virt_pos[data.atom_batch]
    diff = diff - pbc[data.atom_batch] * torch.round(diff)

    weights = (cell[data.atom_batch] @ diff).norm(p=2, dim=1).squeeze()
    norm = scatter(weights, data.atom_batch, reduce="max")[data.atom_batch]
    norm = torch.where(norm == 0.0, torch.ones_like(norm), norm)
    weights = weights / norm

    return weights