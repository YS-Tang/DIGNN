from torch import nn
from ..utils import init_weights

class DIGNN(nn.Module):
    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder
        self.apply(init_weights)
    
    def forward(self, data):
        x_atm, x_bnd, x_ang, x_dih, x_bndI = data.atom, data.BondLength_reo, data.CosAng_reo, data.CosDih_reo, data.BondLengthI_reo # 均为reorgnization数据
        edge_index_bnd, edge_index_ang, edge_index_dih, edge_index_bndI = data.bond_index, data.angle_index, data.dihedral_index, data.bondI_index # 均取完整index
        index_angle_map, index_bond_map, index_dih_map, index_bondI_map = data.AngReo4Ang, data.BndReo4Bnd, data.DihReo4Dih, data.BndReo4BndI
        
        atom_batch = getattr(data, 'atom_batch', None)
        # 预存的批内图数量作为常量 dim_size 传入 decoder,
        # 使 dynamo 将其视为常量, 消除池化处因 unique() 导致的 graph break。
        n_graphs = getattr(data, 'n_graphs', None)
        # pos 仅在 processor 的傅里叶相位分支(use_phase)使用; strip_topo 不删 pos,
        # 力场路径 pos.requires_grad=True 时相位天然接入力求导链。
        pos = getattr(data, 'pos', None)
        # cell 仅在 LES 倒格矢分支(les_use_reciprocal)使用; 晶体存在, 分子为 None -> 自动回退各向同性 k。
        cell = getattr(data, 'cell', None)
        
        e_atm, e_bnd, e_ang, e_dih, e_bndI = self.encoder(x_atm, x_bnd, x_ang, x_dih, x_bndI)
        
        p_atm = self.processor(e_atm, e_bnd, e_ang, e_dih, e_bndI, 
                                edge_index_bnd, edge_index_ang, edge_index_dih, edge_index_bndI,
                                index_angle_map, index_bond_map, index_dih_map, index_bondI_map,
                                atom_batch, n_graphs, pos, cell
                                )
        d_atm = self.decoder(p_atm, atom_batch, dim_size=n_graphs, pos=pos, cell=cell)
        return d_atm