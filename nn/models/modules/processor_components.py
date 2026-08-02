import math

import torch
from torch import nn
from torch_geometric.utils import scatter

from ...utils import MLP


def _build_isotropic_k(n_k: int, lam_min: float, lam_max: float) -> torch.Tensor:
    """构造一组固定的各向同性 k 向量(不可学习), 覆盖 [lam_min, lam_max] 波长范围。

    - 波长 lambda 在对数尺度均匀取样, 对应空间频率 |k| = 2*pi/lambda;
    - 方向取一组近似各向同性的单位向量(基于黄金螺旋在球面上均匀铺点),
      用有限方向近似各向同性 -> 旋转不变性为“近似”而非精确(这是该路线的固有近似)。
    返回 shape [n_k, 3]。
    """
    # 对数均匀的波长 -> 频率幅值
    if n_k == 1:
        lambdas = torch.tensor([math.sqrt(lam_min * lam_max)])
    else:
        t = torch.linspace(0.0, 1.0, n_k)
        lambdas = torch.exp(math.log(lam_min) + t * (math.log(lam_max) - math.log(lam_min)))
    k_mag = 2.0 * math.pi / lambdas                                   # [n_k]
    # 黄金螺旋在单位球面上均匀铺方向
    idx = torch.arange(n_k, dtype=torch.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))                            # 黄金角
    z = 1.0 - 2.0 * (idx + 0.5) / n_k                                 # [-1,1)
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = phi * idx
    dirs = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)  # [n_k,3]
    return dirs * k_mag.unsqueeze(-1)                                 # [n_k,3]


def _build_miller(n_max: int) -> torch.Tensor:
    """生成密勒指数 n(整数组合), 用于倒格矢 k = n · B(B=2π(A⁻¹)ᵀ)。

    只保留 0<|n|²≤n_max² 且去除 ±n 冒余(|S(k)|²=|S(-k)|²), 半减计算量。
    返回 shape [M, 3]。
    """
    millers = []
    seen = set()
    rng = range(-n_max, n_max + 1)
    for a in rng:
        for b in rng:
            for c in rng:
                s = a * a + b * b + c * c
                if s == 0 or s > n_max * n_max:
                    continue
                if (-a, -b, -c) in seen:
                    continue
                seen.add((a, b, c))
                millers.append((a, b, c))
    return torch.tensor(millers, dtype=torch.float32)                # [M, 3]


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

    def __init__(self, dim: int, num_tokens: int = 1, gate_init: float = 0.1,
                 n_k: int = 0,
                 lam_min: float = 3.0, lam_max: float = 15.0,
                 gate_phi_init: float = 0.2,
                 use_reciprocal: bool = False, n_max: int = 2):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        # 相位分支由 n_k>0 开启(取代原 use_phase 布尔): n_k=0 即关闭, 退化为纯 mean-field。
        use_phase = n_k > 0
        self.use_phase = use_phase
        self.use_reciprocal = use_reciprocal
        # 每个 token 的可学习查询向量(决定它关注哪些原子) 与独立 value 变换
        self.query = nn.Parameter(torch.randn(num_tokens, dim) * (dim ** -0.5))
        self.value = nn.Linear(dim, num_tokens * dim)
        # 傅里叶相位分支(可开关): 相位值变换 z_i = phase_value(h_i), 聚合 cos/sin 两路
        # -> 减自身相位得平移不变的 g_phase_{j,k}=sum_i alpha_i z_i cos(k.(r_i-r_j)),
        # 全程实数运算避免复数 autograd。k 有两种来源:
        #   - 分子/回退: 固定各向同性 k(buffer, 全图共享), n_k 个;
        #   - 晶体(use_reciprocal): 每图倒格矢 k=n·B(密勒指数 n, buffer), M 个,
        #     使 exp(ik.r) 对晶格平移严格周期, 消除固定 k 的 minimum-image 歧义。
        if use_phase:
            if use_reciprocal:
                miller = _build_miller(n_max)                         # [M, 3]
                self.register_buffer('miller', miller)
                self.n_k = miller.shape[0]                            # M 覆盖 n_k
            else:
                k_vec = _build_isotropic_k(n_k, lam_min, lam_max)     # [n_k, 3]
                self.register_buffer('k_vec', k_vec)
                self.n_k = n_k
            self.phase_value = nn.Linear(dim, num_tokens * dim)
            # 内层门控: 拼接式融合, gate_phi=0 时相位路输出恒为 0 -> 严格退化到 mean-field。
            self.gate_phi = nn.Parameter(torch.full((1,), float(gate_phi_init)))
            phase_extra = num_tokens * self.n_k
        else:
            self.n_k = 0
            phase_extra = 0
        # 将(原子特征, 拼接的各 token 全局特征[, 相位通道])融合为每个原子的更新量
        self.update = MLP([dim * (1 + num_tokens + phase_extra), dim, dim], act=nn.SiLU())
        self.norm = nn.LayerNorm(dim)
        # LayerScale 式门控: 初始为小正值 gate_init(而非纯 ReZero 的 0)。
        # gate_init=0 时为严格恒等但会压制模块自身梯度导致欠训练/收敛变差;
        # 小正值(如 0.1)既保初始扰动较小(稳定), 又避免梯度被压死(保收敛),
        # 是 CaiT/LayerScale 验证的更优折中。
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))

    def forward(self, h_atm: torch.Tensor, atom_batch: torch.Tensor,
                num_graphs: int = None, pos: torch.Tensor = None,
                cell: torch.Tensor = None) -> torch.Tensor:
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
        feats = [h_atm, g_broadcast]
        if self.use_phase and pos is not None:
            feats.append(self._phase_feature(h_atm, alpha, atom_batch, num_graphs, pos, N, cell))
        delta = self.update(torch.cat(feats, dim=-1))
        return h_atm + self.gate * self.norm(delta)                      # ReZero 门控残差注入

    def _phase_kr(self, pos: torch.Tensor, atom_batch: torch.Tensor,
                  num_graphs: int, cell: torch.Tensor):
        """计算相位 cos(k.r)/sin(k.r), 返回 [N, n_k]。

        - 分子/回退: 固定各向同性 k(全图共享) -> pos @ k_vec.t();
        - 晶体(use_reciprocal 且 cell 存在): 每图倒格矢 k=n·B, k[atom_batch] 广播到原子。
        """
        if self.use_reciprocal and cell is not None:
            A = cell.view(num_graphs, 3, 3)                              # [G,3,3]
            B = 2.0 * math.pi * torch.linalg.inv(A).transpose(-1, -2)    # [G,3,3] 倒格矢(行)
            k = self.miller @ B                                          # [G, M, 3]
            k_atom = k[atom_batch]                                       # [N, M, 3]
            kr = (pos.unsqueeze(1) * k_atom).sum(dim=-1)                 # [N, M]
        else:
            kr = pos @ self.k_vec.t()                                    # [N, n_k]
        return torch.cos(kr), torch.sin(kr)

    def _phase_feature(self, h_atm: torch.Tensor, alpha: torch.Tensor,
                       atom_batch: torch.Tensor, num_graphs: int,
                       pos: torch.Tensor, N: int,
                       cell: torch.Tensor = None) -> torch.Tensor:
        """傅里叶相位摘要(实数实现): 对每个 k 计算带注意力权重的复数结构因子,
        再减自身相位取实部, 得逐原子的距离感知特征。

        z_i = phase_value(h_i)                              [N, num_tokens, dim]
        对每个 k:
          C_k = sum_i alpha_i z_i cos(k.r_i)   (按 batch 隔离)   [G, num_tokens, dim]
          D_k = sum_i alpha_i z_i sin(k.r_i)
          g_phase_{j,k} = C_k[b_j] cos(k.r_j) + D_k[b_j] sin(k.r_j)
        = sum_i alpha_i z_i cos(k.(r_i - r_j)) -> 平移不变, 精确含真实两两距离。
        gate_phi 缩放后拼接; gate_phi=0 时整路为 0, 严格退化。
        """
        z = self.phase_value(h_atm).view(N, self.num_tokens, self.dim)    # [N, T, dim]
        cos_kr, sin_kr = self._phase_kr(pos, atom_batch, num_graphs, cell)  # [N, n_k]
        # z_i 与相位相乘: [N, T, dim, n_k]
        zc = z.unsqueeze(-1) * cos_kr.unsqueeze(1).unsqueeze(1)           # [N, T, dim, n_k]
        zs = z.unsqueeze(-1) * sin_kr.unsqueeze(1).unsqueeze(1)
        # 带注意力权重的结构因子, 按图聚合(严格 batch 隔离)
        aw = alpha.unsqueeze(-1).unsqueeze(-1)                            # [N, T, 1, 1]
        C = scatter(aw * zc, atom_batch, dim=0, dim_size=num_graphs, reduce='sum')  # [G,T,dim,n_k]
        D = scatter(aw * zs, atom_batch, dim=0, dim_size=num_graphs, reduce='sum')
        # 广播回原子并减自身相位
        Cj = C[atom_batch]                                               # [N, T, dim, n_k]
        Dj = D[atom_batch]
        g_phase = Cj * cos_kr.unsqueeze(1).unsqueeze(1) + Dj * sin_kr.unsqueeze(1).unsqueeze(1)
        g_phase = self.gate_phi * g_phase                                # 内层门控, =0 严格退化
        return g_phase.reshape(N, self.num_tokens * self.n_k * self.dim)


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
                 atom_dim: int = None,
                 num_tokens: int = 0,
                 gate_init: float = 0.1,
                 n_k: int = 0,
                 lam_min: float = 3.0,
                 lam_max: float = 15.0,
                 gate_phi_init: float = 0.2,
                 use_reciprocal: bool = False,
                 n_max: int = 2):
        super().__init__()
        self.atm_bnd_imls = atm_bnd_imls
        self.iml_node_only = iml_node_only
        # 全局模块由 num_tokens>0 开启(取代原 use_global_token 布尔); =0 时与原 LCP 完全等价。
        self.use_global_token = num_tokens > 0
        # 每个 IML 层后搭配一个全局交互层(可开关)。
        if self.use_global_token:
            assert atom_dim is not None, "num_tokens>0 时必须提供 atom_dim"
            self.global_layers = nn.ModuleList([
                GlobalInteraction(atom_dim, num_tokens=num_tokens, gate_init=gate_init,
                                  n_k=n_k,
                                  lam_min=lam_min, lam_max=lam_max,
                                  gate_phi_init=gate_phi_init,
                                  use_reciprocal=use_reciprocal, n_max=n_max)
                for _ in range(len(atm_bnd_imls))
            ])
        else:
            self.global_layers = None

    def forward(self,
                h_atm: torch.Tensor,
                h_bndI: torch.Tensor,
                edge_index_bndI: torch.Tensor,
                index_bondI_map,
                atom_batch: torch.Tensor = None,
                n_graphs: int = None,
                pos: torch.Tensor = None,
                cell: torch.Tensor = None) -> torch.Tensor:
        """IML 层前向传播。

        若 use_global_token=True, 则在每个局域交互 IML 层后插入一次全局 virtual node 交互,
        使各局域信息团通过 virtual node 交换全局信息(与 LCP “大范围局域交互”目标一致)。
        n_graphs 由预处理阶段预存的常量(data.n_graphs)传入, 避免图内 int(atom_batch.max())
        触发 Tensor.item() 的 graph break; 未传入时在 GlobalInteraction 内部回退计算。
        pos 仅在 use_phase 分支需要, 用于傅里叶相位求相对距离。
        """
        if self.iml_node_only:
            for idx, atm_bnd_iml in enumerate(self.atm_bnd_imls):
                h_atm = atm_bnd_iml(h_atm, edge_index_bndI, h_bndI[index_bondI_map])
                if self.global_layers is not None:
                    h_atm = self.global_layers[idx](h_atm, atom_batch, n_graphs, pos, cell)
        else:
            for idx, atm_bnd_iml in enumerate(self.atm_bnd_imls):
                h_atm, h_bndI_cplt = atm_bnd_iml(h_atm, edge_index_bndI, h_bndI[index_bondI_map])
                h_bndI = scatter(h_bndI_cplt, index_bondI_map, dim=0, reduce='mean', dim_size=h_bndI.shape[0])
                if self.global_layers is not None:
                    h_atm = self.global_layers[idx](h_atm, atom_batch, n_graphs, pos, cell)
        return h_atm