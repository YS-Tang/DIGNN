import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from ..utils import MLP

# Typing
from torch import Tensor

class GatedGCN_origin(MessagePassing):
    """Gated GCN, also known as edge-gated convolution.
    
    Reference: https://arxiv.org/abs/2003.00982
    
    Different from the original version, in this version, the activation function is SiLU,
    and the normalization is LayerNorm.

    This implementation concatenates the `x_i`, `x_j`, and `e_ij` feature vectors during the edge update.
    """
    def __init__(self, node_dim: int, edge_dim: int, epsilon: float = 1e-5) -> None:
        super().__init__(aggr='add')
        self.W_src  = nn.Linear(node_dim, node_dim)
        self.W_dst  = nn.Linear(node_dim, node_dim)
        self.W_e    = nn.Linear(node_dim*2 + edge_dim, edge_dim)
        self.act    = nn.SiLU()
        self.sigma  = nn.Sigmoid()
        self.norm_x = nn.LayerNorm([node_dim])
        self.norm_e = nn.LayerNorm([edge_dim])
        self.eps    = epsilon

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.W_src.weight); self.W_src.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_dst.weight); self.W_dst.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_e.weight);   self.W_e.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        i, j = edge_index

        # Calculate gated edges
        sigma_e = self.sigma(edge_attr)
        e_sum   = scatter(src=sigma_e, index=i, dim=0)
        e_gated = sigma_e / (e_sum[i] + self.eps)

        # Update the nodes (this utilizes the gated edges)
        out = self.propagate(edge_index, x=x, e_gated=e_gated)
        out = self.W_src(x) + out
        out = x + self.act(self.norm_x(out))

        # Update the edges
        z = torch.cat([x[i], x[j], edge_attr], dim=-1)
        edge_attr = edge_attr + self.act(self.norm_e(self.W_e(z)))

        return out, edge_attr

    def message(self, x_j: Tensor, e_gated: Tensor) -> Tensor:
        return e_gated * self.W_dst(x_j)



class GatedGCN(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, epsilon: float = 1e-5, residual: bool=False) -> None:
        super().__init__(aggr='add')
        self.src_gate  = nn.Linear(node_dim, node_dim) #MLP([node_dim, 128, node_dim], act=nn.SiLU(), chunks=1, dropout=0)
        self.dst_gate  = nn.Linear(node_dim, node_dim)
        self.e_gate = nn.Linear(edge_dim, node_dim)
        
        self.src_update  = nn.Linear(node_dim, node_dim)
        self.dst_update  = nn.Linear(node_dim, node_dim)
        self.e_update = nn.Linear(node_dim, edge_dim)
        
        self.act    = nn.SiLU()
        self.sigma  = nn.Sigmoid()
        self.norm_x = nn.LayerNorm(node_dim)
        self.norm_e = nn.LayerNorm(edge_dim)
        # self.norm_x = nn.BatchNorm1d(node_dim)
        # self.norm_e = nn.BatchNorm1d(edge_dim)
        self.eps    = epsilon
        self.residual = residual
        
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        row, col = edge_index

        # 计算门控系数
        src_gates = self.src_gate(x) # e_src
        dst_gates = self.dst_gate(x) # e_dst
        
        m = src_gates[col] + dst_gates[row] + self.e_gate(edge_attr) # e_nodes + 
        sigma_e = self.sigma(m)
        sigma_sum   = scatter(src=sigma_e, index=row, dim=0, dim_size=x.size(0), reduce='sum')
        e_gated = sigma_e / (sigma_sum[row] + self.eps)

        # 更新节点
        node = self.src_update(x) + self.propagate(edge_index, x=self.dst_update(x), e_gated=e_gated)
        node = self.act(self.norm_x(node))

        # Update the edges
        # edge = torch.cat([x[row], x[col], m], dim=-1)
        edge = self.act(self.norm_e(self.e_update(m)))

        if self.residual:
            node = node + x
            edge = edge + edge_attr
        return node, edge

    def message(self, x_j: Tensor, e_gated: Tensor) -> Tensor:
        return e_gated * x_j
    
