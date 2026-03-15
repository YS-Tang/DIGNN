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