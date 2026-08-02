from abc import ABC
from typing import List

import torch
from torch import nn

from ...utils import MLP
from ...convs.gcn import GatedGCN
from .processor_components import HGC, LCP

class BaseProcessor(ABC, nn.Module):
    """
    图神经网络处理器的基类
    """
    
    def __init__(self,
                 atom_dim: int = 64,
                 bond_dim: int = 64,
                 ang_dim: int = 64,
                 dih_dim: int = 64,
                 bondI_dim: int = 32,
                 pml: int = 2,
                 iml: int = 4, 
                 residual: bool = True,
                 dropout: float = 0.0,
                 init_nn_layer: int=3,
                 use_global_token: bool = False,
                 num_global_tokens: int = 1,
                 global_gate_init: float = 0.1,
                 use_phase: bool = False,
                 n_k: int = 4,
                 lam_min: float = 3.0,
                 lam_max: float = 15.0,
                 gate_phi_init: float = 0.2,
                 phase_use_reciprocal: bool = False,
                 phase_n_max: int = 2):
        super().__init__()
        self.pml = pml
        self.iml = iml
        self.residual = residual
        self.dropout = dropout
        self.atom_dim = atom_dim
        self.use_global_token = use_global_token
        self.num_global_tokens = num_global_tokens
        self.global_gate_init = global_gate_init
        self.use_phase = use_phase
        self.n_k = n_k
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.gate_phi_init = gate_phi_init
        self.phase_use_reciprocal = phase_use_reciprocal
        self.phase_n_max = phase_n_max
        
        self._init_feature_nns(atom_dim, bond_dim, ang_dim, dih_dim, bondI_dim, init_nn_layer)
        self.pml_node_only = None
        self.iml_node_only = None
        
        self.atm_bnd_pmls = None
        self.bnd_ang_pmls = None
        self.ang_dih_pmls = None
        self.atm_bnd_imls = None
    
    def _init_feature_nns(self, atom_dim: int, bond_dim: int, ang_dim: int, dih_dim: int, bondI_dim: int, init_nn_layer: int) -> None:
        """初始化特征变换网络"""
        self.atm_nn = nn.Sequential(
            MLP([atom_dim]*init_nn_layer, act=nn.SiLU(), batch_norm=False, dropout=self.dropout),
            nn.LayerNorm(atom_dim)
        )
        self.bnd_nn = nn.Sequential(
            MLP([bond_dim]*init_nn_layer, act=nn.SiLU(), batch_norm=False, dropout=self.dropout),
            nn.LayerNorm(bond_dim)
        )
        self.ang_nn = nn.Sequential(
            MLP([ang_dim]*init_nn_layer, act=nn.SiLU(), batch_norm=False, dropout=self.dropout),
            nn.LayerNorm(ang_dim)
        )
        self.dih_nn = nn.Sequential(
            MLP([dih_dim]*init_nn_layer, act=nn.SiLU(), batch_norm=False, dropout=self.dropout),
            nn.LayerNorm(dih_dim)
        )
        self.bndI_nn = nn.Sequential(
            MLP([bondI_dim]*init_nn_layer, act=nn.SiLU(), batch_norm=False, dropout=self.dropout),
            nn.LayerNorm(bondI_dim)
        )
    
    def _init_processor_components(self):
        self.hgc = HGC(self.atm_bnd_pmls, self.bnd_ang_pmls, self.ang_dih_pmls, self.pml_node_only)
        self.lcp = LCP(self.atm_bnd_imls, self.iml_node_only,
                       use_global_token=self.use_global_token, atom_dim=self.atom_dim,
                       num_tokens=self.num_global_tokens, gate_init=self.global_gate_init,
                       use_phase=self.use_phase, n_k=self.n_k,
                       lam_min=self.lam_min, lam_max=self.lam_max,
                       gate_phi_init=self.gate_phi_init,
                       use_reciprocal=self.phase_use_reciprocal, n_max=self.phase_n_max)
        
    def forward(self,
                h_atm: torch.Tensor, # reorgnization数据
                h_bnd: torch.Tensor,
                h_ang: torch.Tensor, 
                h_dih: torch.Tensor,
                h_bndI: torch.Tensor,
                
                edge_index_bnd: torch.Tensor, # 完整index
                edge_index_ang: torch.Tensor,
                edge_index_dih: torch.Tensor,
                edge_index_bndI: torch.Tensor,
                
                index_ang_map, # reorgnization数据向完整数据的映射
                index_bond_map, 
                index_dih_map,
                index_bondI_map,
                atom_batch=None,
                n_graphs=None,
                pos=None,
                cell=None) -> torch.Tensor:
        """前向传播
        
        Args:
            h_atm: 原子特征
            h_bnd: 键特征
            h_ang: 角特征
            h_dih: 二面角特征
            h_bndI: IML 层键特征
            edge_index_bnd: 键边索引
            edge_index_ang: 角边索引
            edge_index_dih: 二面角边索引
            edge_index_bndI: IML 层键边索引
            index_ang_map: 角特征映射
            index_bond_map: 键特征映射
            index_dih_map: 二面角特征映射
            index_bondI_map: IML 层键特征映射
            
        Returns:
            torch.Tensor: 更新后的原子特征
        """
        h_atm = self.atm_nn(h_atm)
        h_bnd = self.bnd_nn(h_bnd)
        h_ang = self.ang_nn(h_ang)
        h_dih = self.dih_nn(h_dih)
        
        h_atm = self._pml_forward(h_atm, h_bnd, h_ang, h_dih,
                                  edge_index_bnd, edge_index_ang, edge_index_dih,
                                  index_bond_map, index_ang_map, index_dih_map)
        
        h_atm = self._iml_forward(h_atm, h_bndI, edge_index_bndI, index_bondI_map, atom_batch, n_graphs, pos, cell)
        
        return h_atm
    
    def _pml_forward(self,
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
        return self.hgc(h_atm, h_bnd, h_ang, h_dih,
                        edge_index_bnd, edge_index_ang, edge_index_dih,
                        index_bond_map, index_ang_map, index_dih_map)
    
    def _iml_forward(self,
                   h_atm: torch.Tensor,
                   h_bndI: torch.Tensor,
                   edge_index_bndI: torch.Tensor,
                   index_bondI_map,
                   atom_batch=None,
                   n_graphs=None,
                   pos=None,
                   cell=None) -> torch.Tensor:
        return self.lcp(h_atm, h_bndI, edge_index_bndI, index_bondI_map, atom_batch, n_graphs, pos, cell)



class GCN_Processor(BaseProcessor):
    """基于 GatedGCN 的处理器"""
    
    def __init__(self,
                 atom_dim: int = 64,
                 bond_dim: int = 64,
                 ang_dim: int = 64,
                 dih_dim: int = 64,
                 pml: int = 2,
                 iml: int = 4, 
                 residual: bool = False,
                 dropout: float = 0.0,
                 bondI_dim: int = 32,
                 init_nn_layer: int = 3,
                 use_global_token: bool = False,
                 num_global_tokens: int = 1,
                 global_gate_init: float = 0.1,
                 use_phase: bool = False,
                 n_k: int = 4,
                 lam_min: float = 3.0,
                 lam_max: float = 15.0,
                 gate_phi_init: float = 0.2,
                 phase_use_reciprocal: bool = False,
                 phase_n_max: int = 2):
        super().__init__(atom_dim, bond_dim, ang_dim, dih_dim, bondI_dim, pml, iml, residual, dropout, init_nn_layer, use_global_token, num_global_tokens, global_gate_init, use_phase, n_k, lam_min, lam_max, gate_phi_init, phase_use_reciprocal, phase_n_max)
        
        self.pml_node_only = False
        self.iml_node_only = False
        
        self._init_gcn_layers(atom_dim, bond_dim, ang_dim, dih_dim, bondI_dim)
    
    def _init_gcn_layers(self, atom_dim: int, bond_dim: int, ang_dim: int, dih_dim: int, bondI_dim: int) -> None:
        """初始化 GCN 层"""
        self.atm_bnd_pmls = nn.ModuleList([
            GatedGCN(atom_dim, bond_dim, residual=self.residual) for _ in range(self.pml)
        ])
        self.bnd_ang_pmls = nn.ModuleList([
            GatedGCN(bond_dim, ang_dim, residual=self.residual) for _ in range(self.pml)
        ])
        self.ang_dih_pmls = nn.ModuleList([
            GatedGCN(ang_dim, dih_dim, residual=self.residual) for _ in range(self.pml)
        ])
        self.atm_bnd_imls = nn.ModuleList([
            GatedGCN(atom_dim, bondI_dim, residual=self.residual) for _ in range(self.iml)
        ])
        
        self._init_processor_components()


class GINE_Processor(BaseProcessor):
    """基于 GINE 的处理器"""
    
    def __init__(self,
                 atom_dim: int = 64,
                 bond_dim: int = 64,
                 ang_dim: int = 64,
                 dih_dim: int = 64,
                 pml: int = 2,
                 iml: int = 4, 
                 residual: bool = False,
                 dropout: float = 0.0,
                 bondI_dim: int = 32,
                 gin_nn: List[int] = [64, 128, 64],
                 init_nn_layer: int = 3,
                 use_global_token: bool = False,
                 num_global_tokens: int = 1,
                 global_gate_init: float = 0.1,
                 use_phase: bool = False,
                 n_k: int = 4,
                 lam_min: float = 3.0,
                 lam_max: float = 15.0,
                 gate_phi_init: float = 0.2,
                 phase_use_reciprocal: bool = False,
                 phase_n_max: int = 2):
        super().__init__(atom_dim, bond_dim, ang_dim, dih_dim, bondI_dim, pml, iml, residual, dropout, init_nn_layer, use_global_token, num_global_tokens, global_gate_init, use_phase, n_k, lam_min, lam_max, gate_phi_init, phase_use_reciprocal, phase_n_max)
        
        self.pml_node_only = False
        self.iml_node_only = True
        
        self._init_layers(atom_dim, bond_dim, ang_dim, dih_dim, bondI_dim, gin_nn=gin_nn)
    
    def _init_layers(self, atom_dim: int, bond_dim: int, ang_dim: int, dih_dim: int, bondI_dim: int, gin_nn: List[int] = None) -> None:
        """初始化, 暂只修改IML层交互为GINE"""
        from ...convs.gin import GINE

        self.atm_bnd_pmls = nn.ModuleList([
            GatedGCN(atom_dim, bond_dim, residual=self.residual) for _ in range(self.pml)
        ])
        self.bnd_ang_pmls = nn.ModuleList([
            GatedGCN(bond_dim, ang_dim, residual=self.residual) for _ in range(self.pml)
        ])
        self.ang_dih_pmls = nn.ModuleList([
            GatedGCN(ang_dim, dih_dim, residual=self.residual) for _ in range(self.pml)
        ])
        
        if gin_nn is not None:
            self.atm_bnd_imls = nn.ModuleList([
                GINE(atom_dim, bondI_dim, gin_nn, residual=self.residual) for _ in range(self.iml)
            ])
        else:
            self.atm_bnd_imls = nn.ModuleList([
                GatedGCN(atom_dim, bondI_dim, residual=self.residual) for _ in range(self.iml)
            ])
        
        self._init_processor_components()


class GATv2_Processor(BaseProcessor):
    """基于 GATv2 的处理器"""
    
    def __init__(self,
                 atom_dim: int = 64,
                 bond_dim: int = 64,
                 ang_dim: int = 64,
                 dih_dim: int = 64,
                 pml: int = 2,
                 iml: int = 4, 
                 residual: bool = False,
                 dropout: float = 0.0,
                 bondI_dim: int = 32,
                 gat_heads: int = 1,
                 init_nn_layer: int = 3,
                 use_global_token: bool = False,
                 num_global_tokens: int = 1,
                 global_gate_init: float = 0.1,
                 use_phase: bool = False,
                 n_k: int = 4,
                 lam_min: float = 3.0,
                 lam_max: float = 15.0,
                 gate_phi_init: float = 0.2,
                 phase_use_reciprocal: bool = False,
                 phase_n_max: int = 2):
        super().__init__(atom_dim, bond_dim, ang_dim, dih_dim, bondI_dim, pml, iml, residual, dropout, init_nn_layer, use_global_token, num_global_tokens, global_gate_init, use_phase, n_k, lam_min, lam_max, gate_phi_init, phase_use_reciprocal, phase_n_max)
        
        self.pml_node_only = False
        self.iml_node_only = True
        
        self._init_layers(atom_dim, bond_dim, ang_dim, dih_dim, bondI_dim, gat_heads=gat_heads)
    
    def _init_layers(self, atom_dim: int, bond_dim: int, ang_dim: int, dih_dim: int, bondI_dim: int, gat_heads: int = None) -> None:
        """初始化, 暂只修改IML层交互为GATv2"""
        from torch_geometric.nn import GATv2Conv

        self.atm_bnd_pmls = nn.ModuleList([
            GatedGCN(atom_dim, bond_dim, residual=self.residual) for _ in range(self.pml)
        ])
        self.bnd_ang_pmls = nn.ModuleList([
            GatedGCN(bond_dim, ang_dim, residual=self.residual) for _ in range(self.pml)
        ])
        self.ang_dih_pmls = nn.ModuleList([
            GatedGCN(ang_dim, dih_dim, residual=self.residual) for _ in range(self.pml)
        ])
        
        if gat_heads is not None:
            self.atm_bnd_imls = nn.ModuleList([
                GATv2Conv(in_channels=atom_dim,
                          out_channels=bondI_dim,
                          edge_dim=bondI_dim,
                          heads=gat_heads,
                          residual=self.residual,
                          concat=False) for _ in range(self.iml)
            ])
        else:
            self.atm_bnd_imls = nn.ModuleList([
                GatedGCN(atom_dim, bondI_dim, residual=self.residual) for _ in range(self.iml)
            ])
        
        self._init_processor_components()


class EGAT_Processor(BaseProcessor):
    """基于 EGAT 的处理器"""
    
    def __init__(self,
                 atom_dim: int = 64,
                 bond_dim: int = 64,
                 ang_dim: int = 64,
                 dih_dim: int = 64,
                 pml: int = 2,
                 iml: int = 4, 
                 residual: bool = False,
                 dropout: float = 0.0,
                 bondI_dim: int = 32,
                 egat_heads: int = 4,
                 egat_fc_layers: int = 2,
                 init_nn_layer: int = 3,
                 use_global_token: bool = False,
                 num_global_tokens: int = 1,
                 global_gate_init: float = 0.1,
                 use_phase: bool = False,
                 n_k: int = 4,
                 lam_min: float = 3.0,
                 lam_max: float = 15.0,
                 gate_phi_init: float = 0.2,
                 phase_use_reciprocal: bool = False,
                 phase_n_max: int = 2):
        super().__init__(atom_dim, bond_dim, ang_dim, dih_dim, bondI_dim, pml, iml, residual, dropout, init_nn_layer, use_global_token, num_global_tokens, global_gate_init, use_phase, n_k, lam_min, lam_max, gate_phi_init, phase_use_reciprocal, phase_n_max)
        
        self.pml_node_only = False
        self.iml_node_only = False
        
        self._init_egat_layers(atom_dim, bond_dim, ang_dim, dih_dim, bondI_dim, egat_heads, egat_fc_layers)
    
    def _init_egat_layers(self, atom_dim: int, bond_dim: int, ang_dim: int, dih_dim: int, bondI_dim: int, egat_heads: int, egat_fc_layers: int) -> None:
        """初始化 EGAT 层"""
        from ...convs.egat import EGATs_attention
        
        self.atm_bnd_pmls = nn.ModuleList([
            EGATs_attention(atom_dim, edge_dim=bond_dim, num_heads=egat_heads, num_fc_layers=egat_fc_layers)
            for _ in range(self.pml)
        ])
        self.bnd_ang_pmls = nn.ModuleList([
            EGATs_attention(bond_dim, edge_dim=ang_dim, num_heads=egat_heads, num_fc_layers=egat_fc_layers)
            for _ in range(self.pml)
        ])
        self.ang_dih_pmls = nn.ModuleList([
            EGATs_attention(ang_dim, edge_dim=dih_dim, num_heads=egat_heads, num_fc_layers=egat_fc_layers)
            for _ in range(self.pml)
        ])
        self.atm_bnd_imls = nn.ModuleList([
            EGATs_attention(atom_dim, edge_dim=bondI_dim, num_heads=egat_heads, num_fc_layers=egat_fc_layers)
            for _ in range(self.iml)
        ])
        
        self._init_processor_components()
