import math
from torch import nn
from torch_geometric.utils import scatter

from ..utils import MLP

# Typing
from torch import Tensor

def graph_softmax(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    src_max = scatter(src, index=index, dim=0, dim_size=dim_size, reduce='max')
    out = src - src_max[index]
    out = out.exp()
    out_sum = scatter(out, index=index, dim=0, dim_size=dim_size, reduce='sum') + 1e-12
    out_sum = out_sum[index]
    return out / out_sum

class NodeAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, edge_dim: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim       = dim
        self.num_heads = num_heads
        self.edge_dim  = edge_dim
        self.lin_Q = nn.Linear(dim, dim)
        self.lin_K = nn.Linear(dim, dim)
        self.lin_V = nn.Linear(dim, dim)
        self.lin_O = nn.Linear(dim, dim)
        self.edge_bias = MLP([edge_dim, edge_dim, num_heads], act=nn.SiLU())
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        num_nodes = x.size(0)
        i = edge_index[0]
        j = edge_index[1]

        # Query, key, and value embedding
        d_k = self.dim // self.num_heads
        query = self.lin_Q(x).view(-1, self.num_heads, d_k)[j]
        key   = self.lin_K(x).view(-1, self.num_heads, d_k)[i]
        value = self.lin_V(x).view(-1, self.num_heads, d_k)[i]

        # Multi-head attention
        scores = (query * key).sum(dim=-1) / math.sqrt(d_k) + self.edge_bias(edge_attr)
        alpha = graph_softmax(scores, index=j, dim_size=num_nodes)
        attn_out = (alpha[..., None] * value).view(-1, self.dim)
        attn_out = scatter(attn_out, index=j, dim=0, dim_size=num_nodes)
        return self.lin_O(attn_out)


class GraphormerConv(nn.Module):
    """Graphormer convolution layer.

    Reference: https://arxiv.org/abs/2306.05445
    """
    def __init__(self, dim: int, num_heads: int, ff_dim: list, edge_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.edge_dim  = edge_dim
        self.node_attn = NodeAttention(dim, num_heads, edge_dim)
        self.norm1     = nn.LayerNorm(dim)
        self.norm2     = nn.LayerNorm(dim)
        self.ffn       = MLP([dim, *ff_dim, dim], act=nn.SiLU())
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.norm1(x   + self.node_attn(x, edge_index, edge_attr))
        out = self.norm2(out + self.ffn(out))
        return out
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}, ff_dim={self.ff_dim}, edge_dim={self.edge_dim}'