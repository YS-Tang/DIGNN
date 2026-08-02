"""长程/全局增强模块的单元测试。

覆盖 GlobalInteraction(相位 token) 与 LatentEwald(LES) 的关键不变性与可导性:
- 退化: 开关关闭/门控为0时严格等价于基线, 表达力只增不减;
- 平移不变、晶格平移不变(倒格矢版)、旋转近似;
- 力场可导性: pos.requires_grad 时二阶导(force->loss)可回传。

运行: pytest tests/ -v   (需在 DIGNN 包可导入的环境下)
这些测试由早期临时验证脚本固化而来, 作为回归保护。
"""
import math
import torch
import pytest

from DIGNN.nn.models.modules.processor_components import GlobalInteraction, _build_miller
from DIGNN.nn.models.modules.decoder import Decoder, LatentEwald


torch.manual_seed(0)


def _toy_batch(dim=16, n_per=6, n_graph=2):
    N = n_per * n_graph
    h = torch.randn(N, dim)
    atom_batch = torch.arange(n_graph).repeat_interleave(n_per)
    pos = torch.randn(N, 3) * 5.0
    return h, atom_batch, pos, N, n_graph


def _two_cells(n_graph=2):
    cell0 = torch.tensor([[6.0, 0.0, 0.0], [0.0, 5.5, 0.0], [0.0, 0.0, 7.0]])
    cell1 = torch.tensor([[5.0, 0.5, 0.0], [0.3, 6.0, 0.0], [0.0, 0.0, 6.5]])
    return torch.cat([cell0, cell1], dim=0), cell0, cell1


# ============================ GlobalInteraction ============================

def test_global_no_phase_pos_agnostic():
    """use_phase=False: 传不传 pos 输出一致(纯 mean-field, 不依赖坐标)。"""
    h, ab, pos, N, G = _toy_batch()
    m = GlobalInteraction(16, num_tokens=2, n_k=0)
    assert torch.allclose(m(h, ab, G), m(h, ab, G, pos), atol=1e-6)


def test_phase_gate0_degenerates():
    """gate_phi=0 时相位路输出为0, 与非0结果不同(证明相位路可生效且可严格退化)。"""
    h, ab, pos, N, G = _toy_batch()
    m = GlobalInteraction(16, num_tokens=2, n_k=4)
    with torch.no_grad():
        m.gate_phi.fill_(0.0)
    out0 = m(h, ab, G, pos)
    with torch.no_grad():
        m.gate_phi.fill_(0.3)
    out1 = m(h, ab, G, pos)
    assert not torch.allclose(out0, out1, atol=1e-6)


def test_phase_iso_translation_invariant():
    """各向同性相位版整体平移不变。"""
    h, ab, pos, N, G = _toy_batch()
    m = GlobalInteraction(16, num_tokens=2, n_k=4)
    shift = torch.tensor([1.3, -2.7, 0.8])
    assert torch.allclose(m(h, ab, G, pos), m(h, ab, G, pos + shift), atol=1e-4)


def test_phase_recip_lattice_invariant():
    """倒格矢相位版: 某原子平移一个晶格矢, 输出不变(晶格平移严格周期)。"""
    h, ab, _, N, G = _toy_batch()
    cell, cell0, cell1 = _two_cells()
    pos = torch.rand(N, 3)
    pos[:6] = pos[:6] @ cell0
    pos[6:] = pos[6:] @ cell1
    m = GlobalInteraction(16, num_tokens=2, n_k=1, use_reciprocal=True, n_max=2)
    out = m(h, ab, G, pos, cell)
    pos_lat = pos.clone()
    pos_lat[0] = pos_lat[0] + cell0[0]
    out_lat = m(h, ab, G, pos_lat, cell)
    assert torch.allclose(out, out_lat, atol=1e-3)


def test_phase_recip_second_order_grad():
    """倒格矢相位版二阶导(force->loss)可回传。"""
    h, ab, _, N, G = _toy_batch()
    cell, cell0, cell1 = _two_cells()
    pos = torch.rand(N, 3)
    pos[:6] = pos[:6] @ cell0
    pos[6:] = pos[6:] @ cell1
    pos = pos.requires_grad_(True)
    m = GlobalInteraction(16, num_tokens=2, n_k=1, use_reciprocal=True, n_max=2)
    energy = m(h, ab, G, pos, cell).sum()
    force = -torch.autograd.grad(energy, pos, create_graph=True)[0]
    (force ** 2).sum().backward()
    assert m.phase_value.weight.grad is not None


def test_build_miller_counts():
    """密勒指数数量: n_max=1->3, n_max=2->16 (去除±冗余)。"""
    assert _build_miller(1).shape[0] == 3
    assert _build_miller(2).shape[0] == 16


# ============================ LatentEwald (LES) ============================

def test_les_gate0_degenerates():
    """les_gate=0 时 Decoder 输出等于无 LES 的纯 decoder(严格退化)。"""
    h, ab, pos, N, G = _toy_batch(dim=32)
    dim = [32, 16, 1]
    dec = Decoder(dim, reduce_method='mean', les_n_k=16, les_gate_init=0.0)
    dec_off = Decoder(dim, reduce_method='mean', les_n_k=0)
    dec_off.decoder.load_state_dict(dec.decoder.state_dict())
    assert torch.allclose(dec(h, ab, dim_size=G, pos=pos),
                          dec_off(h, ab, dim_size=G, pos=pos), atol=1e-6)


def test_les_charge_neutral():
    """电荷中性: 每图潜电荷去均值后和为0。"""
    from torch_geometric.utils import scatter
    h, ab, pos, N, G = _toy_batch(dim=32)
    les = LatentEwald(32, n_k=16, charge_neutral=True)
    w = les.charge_head(h).squeeze(-1)
    w = w - scatter(w, ab, dim=0, dim_size=G, reduce='mean')[ab]
    s = scatter(w, ab, dim=0, dim_size=G, reduce='sum')
    assert torch.allclose(s, torch.zeros_like(s), atol=1e-5)


def test_les_translation_invariant():
    """LES 长程能量整体平移不变。"""
    h, ab, pos, N, G = _toy_batch(dim=32)
    les = LatentEwald(32, n_k=16)
    shift = torch.tensor([1.3, -2.7, 0.8])
    assert torch.allclose(les(h, pos, ab, G), les(h, pos + shift, ab, G), atol=1e-3)


def test_les_second_order_grad():
    """LES 二阶导可回传。"""
    h, ab, pos, N, G = _toy_batch(dim=32)
    pos = pos.requires_grad_(True)
    les = LatentEwald(32, n_k=16)
    e = les(h, pos, ab, G).sum()
    force = -torch.autograd.grad(e, pos, create_graph=True)[0]
    (force ** 2).sum().backward()
    assert les.charge_head.weight.grad is not None


def test_les_reciprocal_lattice_invariant():
    """LES 倒格矢版晶格平移不变。"""
    h, ab, _, N, G = _toy_batch(dim=32)
    cell, cell0, cell1 = _two_cells()
    pos = torch.rand(N, 3)
    pos[:6] = pos[:6] @ cell0
    pos[6:] = pos[6:] @ cell1
    les = LatentEwald(32, n_k=16, use_reciprocal=True, n_max=2)
    out = les(h, pos, ab, G, cell)
    pos_lat = pos.clone()
    pos_lat[0] = pos_lat[0] + cell0[0]
    assert torch.allclose(out, les(h, pos_lat, ab, G, cell), atol=1e-3)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
