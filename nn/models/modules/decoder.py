import math

import torch
from torch import nn
from torch_geometric.utils import scatter
from typing import List
from ...utils import MLP


def _build_isotropic_k(n_k: int, lam_min: float, lam_max: float) -> torch.Tensor:
    """构造一组固定各向同性 k 向量(不可学习), 覆盖 [lam_min, lam_max] 波长范围。

    与相位分支同款: 波长对数均匀 -> |k|=2*pi/lambda, 方向用黄金螺旋球面均匀铺点。
    返回 shape [n_k, 3]。LES 用途下 n_k 可取多(每个 k 仅产生 1 个标量结构因子, 极省)。
    """
    if n_k == 1:
        lambdas = torch.tensor([math.sqrt(lam_min * lam_max)])
    else:
        t = torch.linspace(0.0, 1.0, n_k)
        lambdas = torch.exp(math.log(lam_min) + t * (math.log(lam_max) - math.log(lam_min)))
    k_mag = 2.0 * math.pi / lambdas                                   # [n_k]
    idx = torch.arange(n_k, dtype=torch.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))                            # 黄金角
    z = 1.0 - 2.0 * (idx + 0.5) / n_k                                 # [-1,1)
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = phi * idx
    dirs = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)  # [n_k,3]
    return dirs * k_mag.unsqueeze(-1)                                 # [n_k,3]


class LatentEwald(nn.Module):
    """潜Ewald长程能量项(LES, Bingqing Cheng npj Comput Mater 2025 的轻量实现)。

    物理: 每原子学一个标量潜电荷 w_i(无需真实电荷标签); 结构因子
        S(k) = Σ_{i∈g} w_i · exp(i k·r_i)
    倒空间静电能(广延量):
        E_lr,g = Σ_k f(|k|) · |S_g(k)|²,   f(|k|)=exp(-σ²|k|²/2)/|k|²
    实现要点:
    - 全实数运算: |S|² = (Σ w cos)² + (Σ w sin)², 用 cos/sin 双路 scatter, 避免复数 autograd;
    - 严格按 atom_batch 隔离, 每图独立求和;
    - 电荷中性软约束: 对每图减去均值电荷(Σ w_i=0), 规避 |S(k→0)| 发散, 提升 OOD 稳定;
    - k 为固定各向同性 buffer(不可学习), 契合小样本/OOD 与"容量是负债"原则;
    - 可导, pos.requires_grad 时天然接入力求导链, 复杂度 O(N·n_k)。

    返回每图的长程能量 [G, 1](广延量)。是否 per-atom 归一化由上层 Decoder 决定。
    """

    def __init__(self, atom_dim: int, n_k: int = 16,
                 lam_min: float = 3.0, lam_max: float = 15.0,
                 sigma: float = 1.0, charge_neutral: bool = True,
                 use_reciprocal: bool = False, n_max: int = 2):
        super().__init__()
        self.n_k = n_k
        self.charge_neutral = charge_neutral
        self.sigma = sigma
        self.use_reciprocal = use_reciprocal
        # 分子/无 cell 回退路径: 固定各向同性 k(buffer, 不可学习)
        k_vec = _build_isotropic_k(n_k, lam_min, lam_max)             # [n_k, 3]
        self.register_buffer('k_vec', k_vec)
        k_sq = (k_vec ** 2).sum(dim=-1)                               # [n_k]
        # Ewald 倒空间核 f(|k|)=exp(-σ²|k|²/2)/|k|², 固定为 buffer
        self.register_buffer('kernel', torch.exp(-(sigma ** 2) * k_sq / 2.0) / (k_sq + 1e-12))
        # 晶体倒格矢路径: 预生成密勒指数 n(整数组合), k=n·B, B=2π(A⁻¹)ᵀ。
        # 只保留 0<|n|²≤n_max² 且去除 ±n 冒余(|S(k)|²=|S(-k)|²), 半减计算量。
        if use_reciprocal:
            millers = []
            seen = set()
            r = range(-n_max, n_max + 1)
            for a in r:
                for b in r:
                    for c in r:
                        s = a * a + b * b + c * c
                        if s == 0 or s > n_max * n_max:
                            continue
                        if (-a, -b, -c) in seen:      # 去除 ± 冒余
                            continue
                        seen.add((a, b, c))
                        millers.append((a, b, c))
            miller = torch.tensor(millers, dtype=torch.float32)      # [M, 3]
            self.register_buffer('miller', miller)
        # 潜电荷预测头: 每原子 -> 标量
        self.charge_head = nn.Linear(atom_dim, 1)

    def forward(self, h_atm: torch.Tensor, pos: torch.Tensor,
                atom_batch: torch.Tensor, num_graphs: int,
                cell: torch.Tensor = None) -> torch.Tensor:
        w = self.charge_head(h_atm).squeeze(-1)                       # [N] 潜电荷
        if self.charge_neutral:
            # 每图减去均值电荷, 实现 Σ_i w_i = 0 的软中性
            mean_w = scatter(w, atom_batch, dim=0, dim_size=num_graphs, reduce='mean')
            w = w - mean_w[atom_batch]
        if self.use_reciprocal and cell is not None:
            return self._forward_reciprocal(w, pos, atom_batch, num_graphs, cell)
        return self._forward_isotropic(w, pos, atom_batch, num_graphs)

    def _forward_isotropic(self, w, pos, atom_batch, num_graphs):
        """分子/回退路径: 固定各向同性 k(全图共享)。"""
        kr = pos @ self.k_vec.t()                                     # [N, n_k]
        wc = w.unsqueeze(-1) * torch.cos(kr)                          # [N, n_k]
        ws = w.unsqueeze(-1) * torch.sin(kr)
        Sc = scatter(wc, atom_batch, dim=0, dim_size=num_graphs, reduce='sum')  # [G, n_k]
        Ss = scatter(ws, atom_batch, dim=0, dim_size=num_graphs, reduce='sum')
        S_sq = Sc ** 2 + Ss ** 2                                      # |S(k)|² [G, n_k]
        E_lr = (S_sq * self.kernel).sum(dim=-1, keepdim=True)         # [G, 1] 广延量
        return E_lr

    def _forward_reciprocal(self, w, pos, atom_batch, num_graphs, cell):
        """晶体路径: 用每图倒格矢 k, 使 exp(ik·r) 对晶格平移严格周期。

        cell: [G*3, 3] 或 [G, 3, 3], 行向量为晶格矢 a1,a2,a3。
        B = 2π (A⁻¹)ᵀ 使 b_i·a_j = 2πδ_ij; k = n @ B。kernel 随图变化(逐图算)。
        """
        A = cell.view(num_graphs, 3, 3)                              # [G,3,3]
        B = 2.0 * math.pi * torch.linalg.inv(A).transpose(-1, -2)    # [G,3,3] 倒格矢(行)
        k = self.miller @ B                                          # [G, M, 3]
        k_atom = k[atom_batch]                                       # [N, M, 3]
        kr = (pos.unsqueeze(1) * k_atom).sum(dim=-1)                 # [N, M]
        wc = w.unsqueeze(-1) * torch.cos(kr)                         # [N, M]
        ws = w.unsqueeze(-1) * torch.sin(kr)
        Sc = scatter(wc, atom_batch, dim=0, dim_size=num_graphs, reduce='sum')  # [G, M]
        Ss = scatter(ws, atom_batch, dim=0, dim_size=num_graphs, reduce='sum')
        S_sq = Sc ** 2 + Ss ** 2                                     # [G, M]
        k_sq = (k ** 2).sum(dim=-1)                                  # [G, M]
        kernel = torch.exp(-(self.sigma ** 2) * k_sq / 2.0) / (k_sq + 1e-12)
        E_lr = (S_sq * kernel).sum(dim=-1, keepdim=True)            # [G, 1]
        return E_lr


class PoolingModule(nn.Module):
    def __init__(self, reduce_method='mean'):
        super().__init__()
        self.reduce_method = reduce_method

    def forward(self, h_atm, atm_batch=None, dim_size=None):
        if atm_batch is not None:
            # dim_size 优先使用预存的常量(来自 data.n_graphs), 此时 dynamo
            # 视其为常量, pooling 处零 graph break; 未传入时回退 unique().numel()
            # 以保持对 Calculator 等单图推理路径的向后兼容。
            if dim_size is None:
                dim_size = atm_batch.unique().numel()
            h_atm_pooled = scatter(h_atm, atm_batch, dim=0, 
                                   reduce=self.reduce_method, dim_size=dim_size)
            return h_atm_pooled
        else:
            return h_atm.mean(dim=0)

class Decoder(nn.Module):
    def __init__(self, dim: List[int], reduce_method='mean', batch_norm=False, dropout=0.0,
                 les_n_k: int = 0,
                 les_lam_min: float = 3.0, les_lam_max: float = 15.0,
                 les_sigma: float = 1.0, les_charge_neutral: bool = True,
                 les_use_reciprocal: bool = False, les_n_max: int = 2,
                 les_gate_init: float = 0.0) -> None:
        # 如果训练集的label是每原子, 则建议reduced_method使用mean. 如果是总值, 可使用sum
        super().__init__()
        self.dim = dim
        self.pooling = PoolingModule(reduce_method=reduce_method)
        self.decoder = MLP(dim, act=nn.SiLU(), batch_norm=batch_norm, dropout=dropout)
        # LES 长程能量项由 les_n_k>0 开启(取代原 use_les 布尔)。les_gate 初始 0 -> 严格退化;
        # 能量为广延量, 若 label 是每原子(reduce_method='mean'), 则 E_lr 需除以图原子数保量纲一致。
        self.use_les = les_n_k > 0
        if self.use_les:
            # LES 是广延量的静电长程能量项, 仅对能量/广延标量有物理意义;
            # 输出维度须为 1(单能量通道)。用于强度量(带隙/模量)会破坏物理含义。
            assert dim[-1] == 1, (
                f"les_n_k>0 仅适用于单通道能量输出(dim[-1]==1), 当前 dim[-1]={dim[-1]}; "
                "LES 是广延能量项, 不应用于多目标或强度量任务。"
            )
            self.les = LatentEwald(atom_dim=dim[0], n_k=les_n_k,
                                   lam_min=les_lam_min, lam_max=les_lam_max,
                                   sigma=les_sigma, charge_neutral=les_charge_neutral,
                                   use_reciprocal=les_use_reciprocal, n_max=les_n_max)
            # les_gate_init=0 时严格退化; 但 30 epoch 等短训练下门控可能来不及打开,
            # 可调大初值(如 0.1~1.0)给 LES 一个真正生效的机会。
            self.les_gate = nn.Parameter(torch.full((1,), float(les_gate_init)))
            self.les_per_atom = (reduce_method == 'mean')
        else:
            self.les = None

    def forward(self, h_atm, atm_batch=None, dim_size=None, pos=None, cell=None):
        h_pooled = self.pooling(h_atm, atm_batch, dim_size=dim_size)
        out = self.decoder(h_pooled)                                  # [G, out_dim] 短程部分
        if self.use_les and pos is not None and atm_batch is not None:
            num_graphs = dim_size if dim_size is not None else atm_batch.unique().numel()
            E_lr = self.les(h_atm, pos, atm_batch, num_graphs, cell)  # [G, 1] 广延量
            if self.les_per_atom:
                # per-atom label: 长程能量除以图原子数, 与每原子输出量纲一致
                counts = scatter(torch.ones_like(atm_batch, dtype=out.dtype), atm_batch,
                                 dim=0, dim_size=num_graphs, reduce='sum').clamp(min=1.0)
                E_lr = E_lr / counts.unsqueeze(-1)
            out = out + self.les_gate * E_lr
        return out


class Global_Decoder(nn.Module):
    def __init__(self, dim: List[int], reduce_method='mean', batch_norm=False, dropout=0.0) -> None:
        # 如果训练集的label是每原子, 则建议reduced_method使用mean. 如果是总值, 可使用sum
        super().__init__()
        self.dim = dim
        self.decoder = MLP(dim, act=nn.SiLU(), batch_norm=batch_norm, dropout=dropout)
        self.reduce_method = reduce_method

    def forward(self, x_atm, atm_batch, x_bnd, bnd_batch, x_ang, ang_batch, x_dih, dih_batch):
        atm_pooled = scatter(x_atm, atm_batch, dim=0, reduce=self.reduce_method)
        bnd_pooled = scatter(x_bnd, bnd_batch, dim=0, reduce=self.reduce_method)
        ang_pooled = scatter(x_ang, ang_batch, dim=0, reduce=self.reduce_method)
        dih_pooled = scatter(x_dih, dih_batch, dim=0, reduce=self.reduce_method)
        
        out = torch.cat([atm_pooled, bnd_pooled, ang_pooled, dih_pooled], dim=-1)
            
        return self.decoder(out)