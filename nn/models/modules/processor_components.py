import torch
from torch import nn
from torch_geometric.utils import scatter

from ...utils import MLP


class GlobalInteraction(nn.Module):
    """基于 virtual node 的全局信息交互模块。

    设计要点(针对用户之前 12-vnode 方案效果不佳的改进):
    1. 每个图仅 1 个 virtual node, 严格按 atom_batch 隔离, 避免同一 batch
       内不同体系通过共享 vnode 串信息(此为之前方案最可能的失效根因)。
    2. 机制为 pool->广播: vnode = MLP(全图原子特征池化), 再广播回每个原子,
       不复用几何 GatedGCN 算子(其边门控为真实键长设计, 对随机 vnode 边无意义)。
    3. 仅用标量特征, 不涉及坐标 -> 保持平移/旋转不变性, 不破坏能量-力导数关系。
    4. 残差式注入: 初始(未训练时)近似恢复为恒等, 对现有行为扰动最小。

    复杂度 O(N), 兼容 PBC 晶体。
    """

    def __init__(self, dim: int, reduce: str = 'mean'):
        super().__init__()
        self.dim = dim
        self.reduce = reduce
        # 池化后的全局向量变换为 virtual node 表示
        self.to_global = MLP([dim, dim, dim], act=nn.SiLU())
        # 将(原子特征, 全局特征)融合后得到对每个原子的更新量
        self.update = MLP([dim * 2, dim, dim], act=nn.SiLU())
        self.norm = nn.LayerNorm(dim)

    def forward(self, h_atm: torch.Tensor, atom_batch: torch.Tensor) -> torch.Tensor:
        # 按图池化: 得到每个图的 virtual node (严格 batch 隔离)
        num_graphs = int(atom_batch.max()) + 1
        g = scatter(h_atm, atom_batch, dim=0, reduce=self.reduce, dim_size=num_graphs)
        g = self.to_global(g)                       # [num_graphs, dim]
        g_broadcast = g[atom_batch]                 # 广播回每个原子 [N, dim]
        delta = self.update(torch.cat([h_atm, g_broadcast], dim=-1))
        return h_atm + self.norm(delta)             # 残差注入


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
                 atom_dim: int = None):
        super().__init__()
        self.atm_bnd_imls = atm_bnd_imls
        self.iml_node_only = iml_node_only
        self.use_global_token = use_global_token
        # 每个 IML 层后搭配一个全局交互层(可开关); 关闭时与原 LCP 完全等价。
        if use_global_token:
            assert atom_dim is not None, "use_global_token 时必须提供 atom_dim"
            self.global_layers = nn.ModuleList([
                GlobalInteraction(atom_dim) for _ in range(len(atm_bnd_imls))
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