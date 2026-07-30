import torch
from torch import nn
from torch_geometric.utils import scatter
from typing import List
from ...utils import MLP

class PoolingModule(nn.Module):
    def __init__(self, reduce_method='mean'):
        super().__init__()
        self.reduce_method = reduce_method

    def forward(self, h_atm, atm_batch=None, dim_size=None):
        if atm_batch is not None:
            # dim_size 优先使用预存的常量(来自 data.n_graphs), 此时 dynamo
            # 视其为常量, pooling 处零 graph break; 未传入时回退 unique().numel()
            # 以保持对 Calculator 等单图推理路径的向后兼容。
            if dim_size is None:
                dim_size = atm_batch.unique().numel()
            h_atm_pooled = scatter(h_atm, atm_batch, dim=0, 
                                   reduce=self.reduce_method, dim_size=dim_size)
            return h_atm_pooled
        else:
            return h_atm.mean(dim=0)

class Decoder(nn.Module):
    def __init__(self, dim: List[int], reduce_method='mean', batch_norm=False, dropout=0.0) -> None:
        # 如果训练集的label是每原子, 则建议reduced_method使用mean. 如果是总值, 可使用sum
        super().__init__()
        self.dim = dim
        self.pooling = PoolingModule(reduce_method=reduce_method)
        self.decoder = MLP(dim, act=nn.SiLU(), batch_norm=batch_norm, dropout=dropout)

    def forward(self, h_atm, atm_batch=None, dim_size=None):
        h_pooled = self.pooling(h_atm, atm_batch, dim_size=dim_size)
        return self.decoder(h_pooled)


class Global_Decoder(nn.Module):
    def __init__(self, dim: List[int], reduce_method='mean', batch_norm=False, dropout=0.0) -> None:
        # 如果训练集的label是每原子, 则建议reduced_method使用mean. 如果是总值, 可使用sum
        super().__init__()
        self.dim = dim
        self.decoder = MLP(dim, act=nn.SiLU(), batch_norm=batch_norm, dropout=dropout)
        self.reduce_method = reduce_method

    def forward(self, x_atm, atm_batch, x_bnd, bnd_batch, x_ang, ang_batch, x_dih, dih_batch):
        atm_pooled = scatter(x_atm, atm_batch, dim=0, reduce=self.reduce_method)
        bnd_pooled = scatter(x_bnd, bnd_batch, dim=0, reduce=self.reduce_method)
        ang_pooled = scatter(x_ang, ang_batch, dim=0, reduce=self.reduce_method)
        dih_pooled = scatter(x_dih, dih_batch, dim=0, reduce=self.reduce_method)
        
        out = torch.cat([atm_pooled, bnd_pooled, ang_pooled, dih_pooled], dim=-1)
            
        return self.decoder(out)