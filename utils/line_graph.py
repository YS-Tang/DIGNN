import torch
# from torch_geometric.transforms.line_graph import LineGraph
from torch_geometric.utils import coalesce, cumsum, remove_self_loops, scatter



def line_graph(edge_index, num_nodes=None):
    N = num_nodes
    row, col = edge_index
    
    mask = row < col
    row, col = row[mask], col[mask]
    i = torch.arange(row.size(0), dtype=torch.long, device=row.device)
    
    (row, col), i = coalesce(
        torch.stack([
            torch.cat([row, col], dim=0),
            torch.cat([col, row], dim=0)
        ], dim=0),
        torch.cat([i, i], dim=0),
        N,
    )
    # 此时row, col为规范后的输入index(但此时仍然是双向的)，i为index规范后的顺序，用在输出index中

    count = scatter(torch.ones_like(row), row, dim=0,
                    dim_size=N, reduce='sum')
    joints = list(torch.split(i, count.tolist()))

    def generate_grid(x: torch.Tensor) -> torch.Tensor:
        row = x.view(-1, 1).repeat(1, x.numel()).view(-1)
        col = x.repeat(x.numel())
        return torch.stack([row, col], dim=0)

    joints = [generate_grid(joint) for joint in joints]
    joint = torch.cat(joints, dim=1)
    joint, _ = remove_self_loops(joint)
    N = row.size(0) // 2
    joint = coalesce(joint, num_nodes=N) # 规范化

    return joint