import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

# Typing
from torch import Tensor
from typing import List


class MLP(nn.Module):
    """Multi-layer perceptron.
    """
    def __init__(self, dims: List[int], act=None, batch_norm=False, chunks: int = 1, dropout=0.0) -> None:
        """
        Args:
            dims (list of int): Input, hidden, and output dimensions.
            act (activation function, or None): Activation function that
                applies to all but the output layer. For example, 'nn.ReLU()'.
                If None, no activation function is applied.
        """
        super().__init__()
        self.dims = dims
        self.act = act
        self.batch_norm = batch_norm
        self.chunks = chunks  # 控制分段数量
        self.dropout = dropout
        
        num_layers = len(dims)

        layers = []
        for i in range(num_layers-1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if i < num_layers - 2:  # 最后一层不添加BN和act
                if self.batch_norm: 
                    layers += [nn.BatchNorm1d(dims[i+1])]  # 添加BN
                if act is not None:
                    layers += [act, nn.Dropout(dropout)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.chunks > 1 and self.training:  # 仅在训练时启用检查点
            return checkpoint_sequential(self.mlp, self.chunks, x)
        else:
            return self.mlp(x)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(dims={self.dims}, act={self.act})'
    
    
class RBFLayer(nn.Module):
    def __init__(self,
                 start=0.0,
                 end=10.0,
                 period=None,
                 num_gaussians=100,
                 if_decay=False
                 ):
        super(RBFLayer, self).__init__()
        self.start = start
        self.end = end
        self.if_decay = if_decay
        self.period = period
        
        mu = torch.linspace(start, end, num_gaussians).float()
        sigma = torch.ones(num_gaussians).float()
        self.mu = torch.nn.Parameter(mu.view(-1, 1)) # 转为列向量
        # self.register_buffer("mu", mu.view(-1, 1)) # 固定mu不作为参数修改
        self.sigma = torch.nn.Parameter(sigma)  # 作为参数学习
    
    def periodic_distance(self, x, mu):
        diff = torch.abs(x - mu)
        periodic_diff = torch.min(diff, self.period - diff)
        return periodic_diff
        
    def forward(self, x):
        # 高斯 RBF 的公式: exp(-||x - mu||^2 / (2 * sigma^2))
        if self.period is not None:
            dist = self.periodic_distance(x, self.mu.T)
        else:
            dist = x - self.mu.T
        gaussian = torch.exp(-torch.pow(dist, 2) / (2 * self.sigma ** 2))
        if self.if_decay: return gaussian * self.decay(x)
        return gaussian
    
    def decay(self, x):
        # 余弦衰减函数
        cutoff = 0.5 * (1 + torch.cos((x - self.start) * torch.pi / (self.end - self.start)))
        cutoff = torch.where(x <= self.end, cutoff, 0.0)
        return cutoff
    
def init_weights(module: nn.Module, init_type: str = 'xavier', gain: float = 1.0):
    """
    一键初始化所有组件，适配 SiLU + LayerNorm + 残差连接
    
    Args:
        module: 要初始化的模块
        init_type: 初始化类型，可选 'xavier' (默认), 'he', 'trunc_normal'
        gain: 额外的增益系数，用于调整残差网络的初始幅度
    """
    if isinstance(module, nn.Linear):
        if init_type == 'xavier':
            # Xavier/Glorot 初始化（适配 SiLU）
            # SiLU 是平滑激活函数，Xavier 更合适
            nn.init.xavier_normal_(module.weight, gain=gain)
        elif init_type == 'he':
            # He 初始化（适配 SiLU）
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear', gain=gain)
        elif init_type == 'trunc_normal':
            # 截断正态分布（类似 Transformer）
            std = gain * (2.0 / (module.in_features + module.out_features)) ** 0.5
            nn.init.trunc_normal_(module.weight, std=std, a=-2.0 * std, b=2.0 * std)
        else:
            raise ValueError(f"未知的初始化类型: {init_type}")
        
        # 偏置初始化为 0
        if module.bias is not None:
            nn.init.uniform_(module.bias, -0.01, 0.01)
    
    elif isinstance(module, nn.LayerNorm):
        # LayerNorm 保持默认初始化（gamma=1, beta=0）
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Embedding):
        # Embedding 层使用正态分布初始化
        std = gain * (1.0 / module.embedding_dim) ** 0.5
        nn.init.normal_(module.weight, mean=0.0, std=std)