from ..utils import MLP
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GINEConv


class GINE(MessagePassing):
    def __init__(self, node_dim, edge_dim, nn_dim, residual: bool=False):
        super(GINE, self).__init__(aggr='add')
        assert node_dim == nn_dim[0]
        mlp = MLP(nn_dim,act=nn.SiLU(), batch_norm=False)
        mlp.in_features = node_dim
        
        self.conv = GINEConv(nn=mlp, edge_dim=edge_dim)
        self.residual = residual
    
    def forward(self, x, edge_index, edge_attr):
        node = self.conv(x, edge_index, edge_attr)
        if self.residual:
            node = node + x
        return node
        
        
