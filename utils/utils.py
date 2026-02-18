import os
import torch
from torch_geometric.utils import to_undirected, coalesce
from typing import List, Union, Optional

ENV_DEVICE = os.environ.get('DIGNN_ENV') or os.environ.get('DEVICE') or 'cpu'

class DifferentiableClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_val, max_val):
        # 保存截断范围（需转为张量以支持动态值）
        # 前进时进行截断
        ctx.save_for_backward(x)
        ctx.min_val = torch.tensor(min_val, device=x.device, dtype=x.dtype)
        ctx.max_val = torch.tensor(max_val, device=x.device, dtype=x.dtype)
        return torch.clamp(x, ctx.min_val.item(), ctx.max_val.item())

    @staticmethod
    def backward(ctx, grad_output):
        # 梯度直接传递（无论是否被截断）
        # 后退（反向求导）时返回截断前的原梯度
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None  # min_val 和 max_val 的梯度为 None
    

def Triangle_LoopSubgraph_mask(edge_index):
    input_device = edge_index.device
    calc_device = ENV_DEVICE
    
    # 转换为无向图并去重
    edge_index = edge_index.to(calc_device)
    edge_index_reo = coalesce(to_undirected(edge_index))
    
    # 构建稀疏邻接矩阵 (COO格式)
    num_nodes = edge_index_reo.max().item() + 1
    
    adj_sparse = torch.sparse_coo_tensor(
        indices=edge_index_reo,
        values=torch.ones(edge_index_reo.size(1),device=calc_device),
        size=(num_nodes, num_nodes)
    ).coalesce()

    # 稀疏矩阵乘法计算共同邻居数
    tri_counts = torch.sparse.mm(adj_sparse, adj_sparse)
    
    # 生成tri_counts存在的边的哈希键
    tri_indices = tri_counts.indices()
    tri_keys = tri_indices[0] * num_nodes + tri_indices[1]
    
    # 生成输入index的哈希键
    sorted_edges = torch.stack([
        torch.min(edge_index[0], edge_index[1]),
        torch.max(edge_index[0], edge_index[1])
    ])
    current_keys = sorted_edges[0] * num_nodes + sorted_edges[1]
    
    # 使用GPU加速的isin操作
    mask = torch.isin(current_keys, tri_keys)
    
    return mask.to(input_device)


def NeboEdge2OpstEdge(edge_index: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
    """
    高效生成连接两个相邻边的非公共节点组成的边, 即对面的边
    Args:
        edge_index (Tensor): 全图的边索引，形状 [2, num_edges]
        edge_pairs (Tensor): 边对索引，形状 [2, num_pairs]
    Returns:
        new_edges (Tensor): 新边索引，形状 [2, num_pairs]
    """
    input_device = edge_index.device
    calc_device = ENV_DEVICE
    
    edge_index = edge_index.to(calc_device)
    edge_pairs = edge_pairs.to(calc_device)
    
    # 提取对应的两条边并排序 (处理无向性)
    edgeA = edge_index[:, edge_pairs[0]]  # [2, num_pairs]
    edgeB = edge_index[:, edge_pairs[1]]  # [2, num_pairs]
    edgeA_sorted = torch.sort(edgeA, dim=0)[0]  # 每列按升序排列
    edgeB_sorted = torch.sort(edgeB, dim=0)[0]
    
    # 生成四种可能的连接情况掩码
    mask1 = (edgeA_sorted[0] == edgeB_sorted[0])  # 情况1: a1==b1
    mask2 = (edgeA_sorted[0] == edgeB_sorted[1])  # 情况2: a1==b2
    mask3 = (edgeA_sorted[1] == edgeB_sorted[0])  # 情况3: a2==b1
    mask4 = (edgeA_sorted[1] == edgeB_sorted[1])  # 情况4: a2==b2
    
    # 生成候选边 (四种情况)
    candidates = torch.stack([
        torch.stack([edgeA_sorted[1], edgeB_sorted[1]], dim=0),  # 情况1候选
        torch.stack([edgeA_sorted[1], edgeB_sorted[0]], dim=0),  # 情况2候选
        torch.stack([edgeA_sorted[0], edgeB_sorted[1]], dim=0),  # 情况3候选
        torch.stack([edgeA_sorted[0], edgeB_sorted[0]], dim=0)   # 情况4候选
    ], dim=2)  # [2, num_pairs, 4]
    
    # 构建掩码张量并选择最终结果
    masks = torch.stack([mask1, mask2, mask3, mask4], dim=1)  # [num_pairs, 4]
    selected_idx = torch.argmax(masks.float(), dim=1)  # [num_pairs]
    
    # 使用高级索引选取结果
    batch_idx = torch.arange(edge_pairs.size(1), device=calc_device)
    new_edges = candidates[:, batch_idx, selected_idx]
    
    return new_edges.to(input_device)

class AtomIndexMapper:
    def __init__(
        self,
        known_atomic_numbers: Optional[List[int]] = None,
        padding_index: int = 0,
        device: str = 'cpu'
    ):
        """
        原子序号到嵌入索引的映射器。
        
        使用查找表实现向量化映射，支持 GPU 加速。

        Args:
            known_atomic_numbers: 已知原子序号列表（如 [1, 6, 8]），可为 None 延迟设置
            padding_index: 填充位置的索引（默认 0）
            device: 计算设备（'cpu' 或 'cuda'）
        """
        self.padding_index = padding_index
        self.device = device
        self._known_atomic_numbers: List[int] = []
        self._lookup_tensor: Optional[torch.Tensor] = None
        self._num_embeddings: int = padding_index + 1

        if known_atomic_numbers is not None:
            self.set_known_atomic_numbers(known_atomic_numbers)

    @property
    def known_atomic_numbers(self) -> List[int]:
        return self._known_atomic_numbers

    @property
    def lookup_tensor(self) -> torch.Tensor:
        if self._lookup_tensor is None:
            raise RuntimeError("未设置已知原子序号，请先调用 set_known_atomic_numbers()")
        return self._lookup_tensor

    @property
    def num_embeddings(self) -> int:
        return self._num_embeddings

    def set_known_atomic_numbers(self, atomic_numbers: List[int]) -> None:
        """
        设置或更新已知原子序号列表。

        Args:
            atomic_numbers: 原子序号列表
        """
        if not atomic_numbers.any():
            raise ValueError("原子序号列表不能为空")

        unique_nums = torch.unique(
            torch.tensor(atomic_numbers, dtype=torch.long, device=self.device),
            sorted=True
        )
        self._known_atomic_numbers = unique_nums.tolist()

        max_atom_num = int(unique_nums.max().item())
        self._lookup_tensor = torch.full(
            (max_atom_num + 1,),
            -1,
            dtype=torch.long,
            device=self.device
        )

        for idx, atom in enumerate(self._known_atomic_numbers):
            self._lookup_tensor[atom] = idx + self.padding_index + 1

        self._num_embeddings = len(self._known_atomic_numbers) + self.padding_index + 1

    def __call__(self, atom_nums: Union[int, List[int], torch.Tensor]) -> torch.Tensor:
        """
        将原子序号映射为嵌入索引。

        Args:
            atom_nums: 原子序号（标量、列表或张量）

        Returns:
            映射后的索引张量

        Raises:
            ValueError: 遇到未知原子序号时抛出
        """
        input_tensor = torch.as_tensor(atom_nums, dtype=torch.long, device=self.device)
        output = self.lookup_tensor[input_tensor]

        invalid_mask = output == -1
        if invalid_mask.any():
            invalid_atoms = input_tensor[invalid_mask].unique().tolist()
            raise ValueError(
                f"未知原子序号: {invalid_atoms}，允许的原子序号为 {self._known_atomic_numbers}"
            )

        return output

    def get_vocab(self) -> dict:
        """返回原子序号到索引的映射字典。"""
        return {
            atom: idx + self.padding_index + 1
            for idx, atom in enumerate(self._known_atomic_numbers)
        }

    def __repr__(self) -> str:
        return (
            f"AtomIndexMapper(padding={self.padding_index}, "
            f"num_embeddings={self._num_embeddings}, "
            f"num_known_atoms={len(self._known_atomic_numbers)}, "
            f"device={self.device})"
        )
    

from torch_geometric.nn import radius_graph as radius_graph_pyg

def radius_graph(
    pos: torch.Tensor,
    r: float,
    max_num_neighbors: int,
    batch: torch.Tensor = None,
    max_candidate: int = 256,
) -> torch.Tensor:
    """生成距离排序后的半径邻接图（优化版）
    
    Args:
        pos: 原子坐标 [N, 3]
        r: 截断半径 (Å)
        max_num_neighbors: 每个原子保留的最大邻居数
        batch: 批次索引 [N, ]
        max_candidate: 候选邻居数量上限
        
    Returns:
        edge_index: 邻接边 [2, E]
    """
    # 输入校验
    assert pos.dim() == 2 and pos.size(1) == 3, "pos应为[N,3]张量"
    assert r > 0, "截断半径需为正数"

    # Step 1: 生成候选边
    edge_index = radius_graph_pyg(pos, r, max_num_neighbors=max_candidate, batch=batch, loop=False)
    if edge_index.size(1) == 0:  # 无边情况
        return edge_index

    # Step 2: 计算平方距离（节省计算开销）
    row, col = edge_index
    delta = pos[row] - pos[col]
    edge_dist_sq = (delta ** 2).sum(dim=-1)  # [E,]

    # Step 3: 分段排序
    sorted_idx = _segmented_argsort(edge_dist_sq, row, descending=False)
    
    # Step 4: 向量化截断
    unique_nodes, counts = torch.unique(row, return_counts=True)
    cum_counts = torch.cat([torch.tensor([0], device=pos.device), counts.cumsum(0)])
    
    # 生成保留掩码
    starts = cum_counts[:-1]
    ends = torch.minimum(starts + max_num_neighbors, cum_counts[1:])
    pos_indices = torch.arange(len(sorted_idx), device=pos.device)
    
    # 分段查找
    segment_ids = torch.searchsorted(cum_counts, pos_indices, right=True) - 1
    valid_mask = (segment_ids >= 0) & (segment_ids < len(starts))
    valid_ends = torch.where(valid_mask, ends[segment_ids], torch.tensor(0, device=pos.device))
    
    keep_mask = (pos_indices < valid_ends) & valid_mask
    edge_index = edge_index[:, sorted_idx[keep_mask]]

    return edge_index

def _segmented_argsort(
    data: torch.Tensor, 
    segment_ids: torch.Tensor, 
    descending: bool = False
) -> torch.Tensor:
    """分段排序（优化数值稳定性）
    
    Args:
        data: 待排序数据 [E,]
        segment_ids: 分段标识 [E,]
        descending: 是否降序
        
    Returns:
        排序后的全局索引 [E,]
    """
    # 生成分段偏移
    unique_segments, inverse, counts = torch.unique(
        segment_ids, return_inverse=True, return_counts=True)
    offset = torch.cat([torch.zeros(1, device=data.device), counts.cumsum(0)[:-1]])

    # 计算稳定的排序键
    scaling = data.max() - data.min() + 1e-6  # 防零除
    sort_key = data + offset[inverse] * scaling  # 确保跨段不重叠

    return sort_key.argsort(descending=descending)