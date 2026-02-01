import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from ..utils import MLP

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


class MultiHeadGatedGCN_test(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, heads: int=4, epsilon: float = 1e-6, residual: bool=False) -> None:
        super().__init__(aggr='add')
        assert node_dim % heads == 0
        self.heads = heads
        self.head_dim = node_dim // heads
        
        # 门控变换
        self.src_gate  = nn.Linear(node_dim, node_dim) #MLP([node_dim, 128, node_dim], act=nn.SiLU(), chunks=1, dropout=0)
        self.dst_gate  = nn.Linear(node_dim, node_dim)
        self.e_gate = nn.Linear(edge_dim, node_dim)
        
        # 特征更新
        self.src_update  = nn.Linear(node_dim, node_dim)
        self.dst_update  = nn.Linear(node_dim, node_dim)
        self.e_update = nn.Linear(node_dim, edge_dim)
        
        # 注意力参数
        self.attn_weights = nn.Parameter(torch.Tensor(self.heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_weights)
        
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

        # 步骤1：多头拆分 (将特征拆分为 [..., heads, head_dim])计算门控系数
        src_gates = self.src_gate(x).view(-1, self.heads, self.head_dim)  # [N_node, head, head_dim]
        dst_gates = self.dst_gate(x).view(-1, self.heads, self.head_dim)  # [N_node, head, head_dim]
        e_gates = self.e_gate(edge_attr).view(-1, self.heads, self.head_dim)  # [N_edge, head, head_dim]
        
        # 步骤2：多头门控计算
        m = src_gates[col] + dst_gates[row] + e_gates # [N_edge, head, head_dim]
        attn_scores = torch.einsum('ehd,hd->eh', m, self.attn_weights) # [N_edge, head]
        sigma_e = self.sigma(attn_scores) # [N_edge, head]
        
        # 步骤3：多头聚合 (每个头独立计算)
        sigma_sum   = scatter(src=sigma_e,
                              index=row,
                              dim=0,
                              dim_size=x.size(0),
                              reduce='sum') # [N_node, head]
        e_gated = sigma_e / (sigma_sum[row] + self.eps) # [N_edge, head]

        # 步骤4：消息传递（按头加权）
        propagated = self.propagate(edge_index, x=x, e_gated=e_gated)  # [N_node, node_dim]
        node = self.src_update(x) + propagated
        node = self.act(self.norm_x(node))
        
        edge = self.e_update(m.view(m.size(0), self.heads*self.head_dim)) # [N_edge, edge_dim]
        edge = self.act(self.norm_e(edge))

        if self.residual:
            node = node + x
            edge = edge + edge_attr
            
        return node, edge

    def message(self, x_j: Tensor, e_gated: Tensor) -> Tensor:
        x_j_dst = self.dst_update(x_j).view(-1, self.heads, self.head_dim)  # [N_edge, head, head_dim]
        return (x_j_dst * e_gated.unsqueeze(-1)).view(-1, self.heads * self.head_dim) # [N_node, head, head_dim]


class MultiHeadGatedGCN_test_v2(MessagePassing):
    # 增加query
    def __init__(self, node_dim: int, edge_dim: int, heads: int=4, epsilon: float = 1e-6, residual: bool=False) -> None:
        super().__init__(aggr='add')
        assert node_dim % heads == 0
        self.heads = heads
        self.head_dim = node_dim // heads
        
        # 门控变换
        self.src_gate  = nn.Linear(node_dim, node_dim) #MLP([node_dim, 128, node_dim], act=nn.SiLU(), chunks=1, dropout=0)
        self.dst_gate  = nn.Linear(node_dim, node_dim)
        self.e_gate = nn.Linear(edge_dim, node_dim)
        
        # 特征更新
        self.src_update  = nn.Linear(node_dim, node_dim)
        self.dst_update  = nn.Linear(node_dim, node_dim)
        self.e_update = nn.Linear(node_dim, edge_dim)
        
        # 注意力参数
        self.attn_weights = nn.Parameter(torch.Tensor(self.heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_weights)
        
        self.act    = nn.SiLU()
        self.sigma  = nn.Sigmoid()
        self.norm_x = nn.LayerNorm(node_dim)
        self.norm_e = nn.LayerNorm(edge_dim)
        # self.norm_x = nn.BatchNorm1d(node_dim)
        # self.norm_e = nn.BatchNorm1d(edge_dim)
        self.eps    = epsilon
        
        self.residual = residual

        self.q_proj = nn.Linear(self.head_dim, self.head_dim)
        self.k_proj = nn.Linear(self.head_dim, self.head_dim)


    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        row, col = edge_index

        # 步骤1：多头拆分 (将特征拆分为 [..., heads, head_dim])计算门控系数
        src_gates = self.src_gate(x).view(-1, self.heads, self.head_dim)  # [N_node, head, head_dim]
        dst_gates = self.dst_gate(x).view(-1, self.heads, self.head_dim)  # [N_node, head, head_dim]
        e_gates = self.e_gate(edge_attr).view(-1, self.heads, self.head_dim)  # [N_edge, head, head_dim]
        
        # 步骤2：多头门控计算
        m = src_gates[col] + dst_gates[row] + e_gates # [N_edge, head, head_dim]

        query = self.q_proj(m[col])  # 或基于目标节点
        key = self.k_proj(m)
        attn_scores = torch.einsum('ehd,ehd->eh', query, key) / torch.sqrt(self.head_dim)

        sigma_e = self.sigma(attn_scores) # [N_edge, head]
        
        # 步骤3：多头聚合 (每个头独立计算)
        sigma_sum   = scatter(src=sigma_e,
                              index=row,
                              dim=0,
                              dim_size=x.size(0),
                              reduce='sum') # [N_node, head]
        e_gated = sigma_e / (sigma_sum[row] + self.eps) # [N_edge, head]

        # 步骤4：消息传递（按头加权）
        propagated = self.propagate(edge_index, x=x, e_gated=e_gated)  # [N_node, node_dim]
        node = self.src_update(x) + propagated
        node = self.act(self.norm_x(node))
        
        edge = self.e_update(m.view(m.size(0), self.heads*self.head_dim)) # [N_edge, edge_dim]
        edge = self.act(self.norm_e(edge))

        if self.residual:
            node = node + x
            edge = edge + edge_attr
            
        return node, edge

    def message(self, x_j: Tensor, e_gated: Tensor) -> Tensor:
        x_j_dst = self.dst_update(x_j).view(-1, self.heads, self.head_dim)  # [N_edge, head, head_dim]
        return (x_j_dst * e_gated.unsqueeze(-1)).view(-1, self.heads * self.head_dim) # [N_node, head, head_dim]