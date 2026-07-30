import torch
from torch import nn
from torch_geometric.utils import scatter

from ...utils import MLP


class GlobalInteraction(nn.Module):
    """基于 virtual node 的全局信息交互模块。

    设计要点(针对用户之前 12-vnode 方案效果不佳的改进):
    1. 严格按 atom_batch 隔离, 避免同一 batch 内不同体系串信息。
    2. 支持 num_tokens 个 virtual node, 每个 token 拥有独立的查询向量与 value 变换,
       通过注意力池化(而非单一 mean)使各 token 关注不同原子子集, 避免多 token
       学到相同信息(类似多头注意力 / Set Transformer 的 PMA)。num_tokens=1 时退化
       为单个学习加权池化。
    3. 无显式边、无自由 vnode 参数: vnode 每层由全图原子特征当场聚合得到,
       是数据驱动的有意义全局摘要, 无需随机初始化。
    4. 仅用标量特征, 不涉坐标 -> 保持平移/旋转不变性, 不破坏能量-力导数关系。
    5. 残差式注入, 初始近似恒等, 对现有行为扰动最小。复杂度 O(N), 兼容 PBC。
    """

    def __init__(self, dim: int, num_tokens: int = 1):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        # 每个 token 的可学习查询向量(决定它关注哪些原子) 与独立 value 变换
        self.query = nn.Parameter(torch.randn(num_tokens, dim) * (dim ** -0.5))
        self.value = nn.Linear(dim, num_tokens * dim)
        # 将(原子特征, 拼接的各 token 全局特征)融合为每个原子的更新量
        self.update = MLP([dim * (1 + num_tokens), dim, dim], act=nn.SiLU())
        self.norm = nn.LayerNorm(dim)

    def forward(self, h_atm: torch.Tensor, atom_batch: torch.Tensor,
                num_graphs: int = None) -> torch.Tensor:
        if num_graphs is None:
            num_graphs = int(atom_batch.max()) + 1
        N = h_atm.size(0)
        # 每个 token 对各原子的打分 [N, num_tokens]
        scores = (h_atm @ self.query.t()) / (self.dim ** 0.5)
        # 按图内原子做 softmax(严格 batch 隔离): 减去段内最大值后 exp 再归一化
        smax = scatter(scores, atom_batch, dim=0, dim_size=num_graphs, reduce='max')
        expv = (scores - smax[atom_batch]).exp()
        denom = scatter(expv, atom_batch, dim=0, dim_size=num_graphs, reduce='sum') + 1e-12
        alpha = expv / denom[atom_batch]                                  # [N, num_tokens]
        # value 拆分为每个 token 独立的变换 [N, num_tokens, dim]
        val = self.value(h_atm).view(N, self.num_tokens, self.dim)
        # 注意力加权池化 -> 每图每 token 一个全局向量 [num_graphs, num_tokens, dim]
        weighted = alpha.unsqueeze(-1) * val
        g = scatter(weighted, atom_batch, dim=0, dim_size=num_graphs, reduce='sum')
        g_broadcast = g[atom_batch].reshape(N, self.num_tokens * self.dim)  # 广播回原子
        delta = self.update(torch.cat([h_atm, g_broadcast], dim=-1))
        return h_atm + self.norm(delta)                                  # 残差注入


class HGC(nn.Module):
    def __init__(self,
                 atm_bnd_pmls: nn.ModuleList,
                 bnd_ang_pmls: nn.ModuleList,
                 ang_dih_pmls: nn.ModuleList,
                 pml_node_only: bool = False):
        super().__init__()
        self.atm_bnd_pmls = atm_bnd_pmls
        self.bnd_ang_pmls = bnd_ang_pmls
        self.ang_dih_pmls = ang_dih_pmls
        self.pml_node_only = pml_node_only
        self.pml = len(atm_bnd_pmls)
    
    def forward(self,
                h_atm: torch.Tensor,
                h_bnd: torch.Tensor,
                h_ang: torch.Tensor,
                h_dih: torch.Tensor,
                edge_index_bnd: torch.Tensor,
                edge_index_ang: torch.Tensor,
                edge_index_dih: torch.Tensor,
                index_bond_map,
                index_ang_map,
                index_dih_map) -> torch.Tensor:
        """PML 层前向传播"""
        if self.pml_node_only:
            for ang_dih_pml, bnd_ang_pml, atm_bnd_pml in zip(self.ang_dih_pmls, 
                                                            self.bnd_ang_pmls,
                                                            self.atm_bnd_pmls):
                h_ang = ang_dih_pml(h_ang, edge_index_dih, h_dih[index_dih_map])
                h_bnd = bnd_ang_pml(h_bnd, edge_index_ang, h_ang[index_ang_map])
                h_atm = atm_bnd_pml(h_atm, edge_index_bnd, h_bnd[index_bond_map])
        else:
            for ang_dih_pml, bnd_ang_pml, atm_bnd_pml in zip(self.ang_dih_pmls, 
                                                            self.bnd_ang_pmls,
                                                            self.atm_bnd_pmls):
                h_ang, h_dih_cplt = ang_dih_pml(h_ang, edge_index_dih, h_dih[index_dih_map])
                if self.pml > 1:
                    h_dih = scatter(h_dih_cplt, index_dih_map, dim=0, reduce='mean', dim_size=h_dih.shape[0])
                    
                h_bnd, h_ang_cplt = bnd_ang_pml(h_bnd, edge_index_ang, h_ang[index_ang_map])
                if self.pml > 1:
                    h_ang = scatter(h_ang_cplt, index_ang_map, dim=0, reduce='mean', dim_size=h_ang.shape[0])
                    
                h_atm, h_bnd_cplt = atm_bnd_pml(h_atm, edge_index_bnd, h_bnd[index_bond_map])
                if self.pml > 1:
                    h_bnd = scatter(h_bnd_cplt, index_bond_map, dim=0, reduce='mean', dim_size=h_bnd.shape[0])

        return h_atm


class LCP(nn.Module):
    def __init__(self,
                 atm_bnd_imls: nn.ModuleList,
                 iml_node_only: bool = False,
                 use_global_token: bool = False,
                 atom_dim: int = None,
                 num_tokens: int = 1):
        super().__init__()
        self.atm_bnd_imls = atm_bnd_imls
        self.iml_node_only = iml_node_only
        self.use_global_token = use_global_token
        # 每个 IML 层后搭配一个全局交互层(可开关); 关闭时与原 LCP 完全等价。
        if use_global_token:
            assert atom_dim is not None, "use_global_token 时必须提供 atom_dim"
            self.global_layers = nn.ModuleList([
                GlobalInteraction(atom_dim, num_tokens=num_tokens) for _ in range(len(atm_bnd_imls))
            ])
        else:
            self.global_layers = None

    def forward(self,
                h_atm: torch.Tensor,
                h_bndI: torch.Tensor,
                edge_index_bndI: torch.Tensor,
                index_bondI_map,
                atom_batch: torch.Tensor = None) -> torch.Tensor:
        """IML 层前向传播。

        若 use_global_token=True, 则在每个局域交互 IML 层后插入一次全局 virtual node 交互,
        使各局域信息团通过 virtual node 交换全局信息(与 LCP “大范围局域交互”目标一致)。
        """
        if self.iml_node_only:
            for idx, atm_bnd_iml in enumerate(self.atm_bnd_imls):
                h_atm = atm_bnd_iml(h_atm, edge_index_bndI, h_bndI[index_bondI_map])
                if self.global_layers is not None:
                    h_atm = self.global_layers[idx](h_atm, atom_batch)
        else:
            for idx, atm_bnd_iml in enumerate(self.atm_bnd_imls):
                h_atm, h_bndI_cplt = atm_bnd_iml(h_atm, edge_index_bndI, h_bndI[index_bondI_map])
                h_bndI = scatter(h_bndI_cplt, index_bondI_map, dim=0, reduce='mean', dim_size=h_bndI.shape[0])
                if self.global_layers is not None:
                    h_atm = self.global_layers[idx](h_atm, atom_batch)
        return h_atm