from torch import nn
from ..utils import init_weights

class DIGNN(nn.Module):
    def __init__(self, encoder, processor, decoder, global_processor):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder
        self.processor.lcp.global_processor = global_processor
        self.apply(init_weights)
    
    def forward(self, data):
        x_atm, x_bnd, x_ang, x_dih, x_bndI = data.atom, data.BondLength_reo, data.CosAng_reo, data.CosDih_reo, data.BondLengthI_reo # 均为reorgnization数据
        edge_index_bnd, edge_index_ang, edge_index_dih, edge_index_bndI = data.bond_index, data.angle_index, data.dihedral_index, data.bondI_index # 均取完整index
        index_angle_map, index_bond_map, index_dih_map, index_bondI_map = data.AngReo4Ang, data.BndReo4Bnd, data.DihReo4Dih, data.BndReo4BndI
        
        atom_batch = getattr(data, 'atom_batch', None)
        
        e_atm, e_bnd, e_ang, e_dih, e_bndI = self.encoder(x_atm, x_bnd, x_ang, x_dih, x_bndI)
        
        self.processor.lcp.global_processor.preprocess(e_atm, atom_batch)
        p_atm = self.processor(e_atm, e_bnd, e_ang, e_dih, e_bndI, 
                                edge_index_bnd, edge_index_ang, edge_index_dih, edge_index_bndI,
                                index_angle_map, index_bond_map, index_dih_map, index_bondI_map
                                )
        d_atm = self.decoder(p_atm, atom_batch)
        return d_atm