import torch
from torch import nn
from torch_geometric.utils import scatter

from ..utils import MLP

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


class EdgeProcessor(nn.Module):
    def __init__(self, dims: List[int], batch_norm=False, chunk:int=1, dropout=0.0) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(MLP(dims, 
                                          act=nn.SiLU(), 
                                          batch_norm=batch_norm, 
                                          chunks=chunk, 
                                          dropout=dropout), 
                                      nn.LayerNorm(dims[-1]),
                                      )

    def forward(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        out = torch.cat([x_i, x_j, edge_attr], dim=-1)
        out = self.edge_mlp(out)
        return edge_attr + out


class NodeProcessor(nn.Module):
    def __init__(self, dims: List[int], batch_norm=False, dropout=0.0) -> None:
        super().__init__()
        self.node_mlp = nn.Sequential(MLP(dims, 
                                          act=nn.SiLU(),
                                          batch_norm=batch_norm, 
                                          dropout=dropout), 
                                      nn.LayerNorm(dims[-1]), 
                                      )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        j   = edge_index[1] # 向egde终点传递信息
        out = scatter(edge_attr, index=j, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=-1)
        out = self.node_mlp(out)
        return x + out


class UniDire_MeshGraphNetsConv(nn.Module):
    """MeshGraphNets convolution/processor operation.

    Reference: https://arxiv.org/pdf/2010.03409v4.pdf

    Notes:
        Different from the original formulation, this version does not account for multiple edge sets.
        TODO: allow for multiple edge sets.
    已做删改: 单向传递, 二面角->键角->键长->原子
    """
    def __init__(self, dim: List[int]=[256,128,128,128], batch_norm=False, dropout=0.0) -> None:
        super().__init__()
        self.node_processor = NodeProcessor(dim,batch_norm=batch_norm, dropout=dropout)
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.node_processor(x, edge_index, edge_attr)
        return x, edge_attr

    def extra_repr(self) -> str:
        return f'node_dim={self.dim}'
    

class BiDire_MeshGraphNetsConv(nn.Module):
    """MeshGraphNets convolution/processor operation.

    Reference: https://arxiv.org/pdf/2010.03409v4.pdf

    Notes:
        Different from the original formulation, this version does not account for multiple edge sets.
        TODO: allow for multiple edge sets.
    双向传递, 允许反复传递信息, 但并非diffusion
    """
    def __init__(self, node_dim: int, edge_dim: int, no_input_dim: List[int], batch_norm=False, EdgeProcessor_chunk:int=1, dropout=0.0) -> None:
        super().__init__()
        self.node_processor = NodeProcessor([node_dim + edge_dim] + no_input_dim,
                                            batch_norm=batch_norm, 
                                            dropout=dropout)
        
        self.edge_processor = EdgeProcessor([node_dim*2 + edge_dim] + no_input_dim,
                                            batch_norm=batch_norm, 
                                            chunk=EdgeProcessor_chunk, 
                                            dropout=dropout)
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        new_x = self.node_processor(x, edge_index, edge_attr)
        new_edge_attr = self.edge_processor(x[edge_index[0]], x[edge_index[1]], edge_attr)
        return new_x, new_edge_attr