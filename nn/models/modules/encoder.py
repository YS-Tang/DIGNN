import torch
from torch import nn
from ...utils import RBFLayer

class Encoder(nn.Module):
    def __init__(self, 
                 num_species: int = 30, 
                 atom_dim: int = 64,
                 bond_dim: int = 64,
                 ang_dim: int = 64,
                 dih_dim: int = 64,                 
                 pml_rcut: float = 5.0,
                 bondI_dim: int=64,
                 iml_rcut=8.0,
                 ) -> None:
        super().__init__()
                
        self.embed_atm = nn.Embedding(num_species, atom_dim, padding_idx=0)
        self.embed_bnd = RBFLayer(start=0.0, end=pml_rcut, num_gaussians=bond_dim, if_decay=True)
        self.embed_ang = RBFLayer(start=-1, end=1, num_gaussians=ang_dim, if_decay=False)
        self.embed_dih = RBFLayer(start=-1, end=1, num_gaussians=dih_dim, if_decay=False)
        
        self.embed_bnd2 = RBFLayer(start=0.0, end=iml_rcut, num_gaussians=bondI_dim, if_decay=True)
        
    def forward(self,  
                x_atm: torch.Tensor,
                x_bnd: torch.Tensor,
                x_ang: torch.Tensor,
                x_dih: torch.Tensor,
                x_bndI: torch.Tensor):
        h_atm = self.embed_atm(x_atm.long().view(-1))
        h_bnd = self.embed_bnd(x_bnd.view(-1,1))
        h_ang = self.embed_ang(x_ang.view(-1,1))
        h_dih = self.embed_dih(x_dih.view(-1,1))
        h_bndI = self.embed_bnd2(x_bndI.view(-1,1))
        return h_atm, h_bnd, h_ang, h_dih, h_bndI