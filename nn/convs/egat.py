import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, Parameter
from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax as tg_softmax
from torch_scatter import scatter


class EGATs_attention(MessagePassing):
    def __init__(self, dim, activation='silu', use_batch_norm=False, dropout_rate=0.0, 
                 num_heads=4, add_bias=True, num_fc_layers=2, edge_dim=None, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.dim = dim
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.add_bias = add_bias
        self.num_fc_layers = num_fc_layers
        self.edge_dim = edge_dim if edge_dim is not None else dim
        
        self.W_e = nn.Linear(edge_dim, dim)

        self.bn1 = nn.BatchNorm1d(num_heads) if use_batch_norm else None
        self.W = Parameter(torch.Tensor(dim * 2, num_heads * dim))
        self.att = Parameter(torch.Tensor(1, num_heads, 2 * dim))
        if add_bias:
            self.bias = Parameter(torch.Tensor(dim))
        else:
            self.bias = None

        self.edge_transform = nn.Linear(1, self.edge_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.edge_transform.weight)
        if self.edge_transform.bias is not None:
            nn.init.zeros_(self.edge_transform.bias)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), device=x.device)
        edge_attr = self.W_e(edge_attr)
        out, edge_attr_updated = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return out, edge_attr_updated

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        out_i = torch.cat([x_i, edge_attr], dim=-1)
        out_j = torch.cat([x_j, edge_attr], dim=-1)

        act_func = getattr(F, self.activation)
        out_i = act_func(torch.matmul(out_i, self.W))
        out_j = act_func(torch.matmul(out_j, self.W))

        out_i = out_i.view(-1, self.num_heads, self.dim)
        out_j = out_j.view(-1, self.num_heads, self.dim)

        alpha = (torch.cat([out_i, out_j], dim=-1) * self.att).sum(dim=-1)
        if self.use_batch_norm and self.bn1 is not None:
            alpha = self.bn1(alpha)
        alpha = act_func(alpha)
        alpha = tg_softmax(alpha, edge_index_i)
        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        alpha_avg = alpha.mean(dim=1, keepdim=True)
        self.edge_attr_updated = self.edge_transform(alpha_avg)

        out_j = (out_j * alpha.view(-1, self.num_heads, 1)).transpose(0, 1)
        return out_j

    def update(self, aggr_out, edge_attr):
        out = aggr_out.mean(dim=0)
        if self.bias is not None:
            out = out + self.bias
        return out, self.edge_attr_updated