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

    # 向量化分段笛卡尔积: 对每个节点的关联边 id 集合做全配对,
    # 等价于原先逐节点 generate_grid + list 拼接, 但消除了 Python 循环。
    sizes = count                      # 每个节点关联的(有向)边数, 即每段大小
    seg_start = cumsum(sizes)[:-1]     # 每段在 i 中的起始偏移
    sq = sizes * sizes                 # 每段输出的配对数 n^2
    out_start = cumsum(sq)[:-1]        # 每段在输出中的起始偏移
    total = int(sq.sum())

    if total == 0:
        return torch.empty(2, 0, dtype=torch.long, device=row.device)

    # 为每个输出列定位所属段, 并还原段内的 (a, b) 局部下标
    seg = torch.repeat_interleave(torch.arange(sizes.size(0), device=row.device), sq)
    k = torch.arange(total, device=row.device) - out_start[seg]
    n = sizes[seg]
    a = k // n                         # 行方向: x 每元素重复 n 次
    b = k - a * n                      # 列方向: x 整体平铺 n 次
    base = seg_start[seg]
    joint = torch.stack([i[base + a], i[base + b]], dim=0)

    joint, _ = remove_self_loops(joint)
    N = row.size(0) // 2
    joint = coalesce(joint, num_nodes=N) # 规范化

    return joint