import torch
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import scatter
from .utils import scatter_softmax

def init_xavier_uniform(
    module: torch.nn.Module, zero_bias: bool = True
) -> None:
    r"""initialize (in place) weights of the input module using xavier uniform.
    Works only on `torch.nn.Linear` at the moment and the bias are set to 0
    by default.

    Parameters
    ----------
    module:
        a torch module
    zero_bias:
        If True, the bias will be filled with zeroes. If False,
        the bias will be filled according to the torch.nn.Linear
        default distribution:

        .. math:

            \text{Uniform}(-\sqrt{k}, sqrt{k}) ; \quad \quad k = \frac{1}{\text{in_features}}

    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None and zero_bias == True:
            torch.nn.init.constant_(module.bias, 0.0)
            
class AggregationBlock(torch.nn.Module):
    """
    Attention-based aggregation block that collects messages from neighbor nodes
    and produces updated node embeddings.

    Initialization parameters
    -------------------------
    in_channels (int): real_hidden_channels
    out_channels (int): virt_hidden_channels (node)
    n_heads (int): num_virt_heads
    basis_dim (int): virt_basis_dim (edge)
    activation (torch.nn.Module): non-linear activation used after aggregation
    """

    def __init__(
        self,
        in_channels: int,  # real_hidden_channels
        out_channels: int,  # virt_hidden_channels (node)
        n_heads: int, # num_virt_heads
        basis_dim: int,  # virt_basis_dim (edge)
        activation: torch.nn.Module=torch.nn.SiLU(),
        **kwargs,
    ):
        super().__init__()

        if in_channels % n_heads != 0:
            raise ValueError(
                "The number of input attention channels must be divisible by the number of heads"
            )

        if out_channels % n_heads != 0:
            raise ValueError(
                "The number of output attention channels must be divisible by the number of heads"
            )

        self.n_heads = n_heads
        self.channels = out_channels
        self.hidden_channels = out_channels // self.n_heads

        self.lin_Q = torch.nn.Linear(out_channels, out_channels, bias=False)
        self.lin_K = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.lin_V = torch.nn.Linear(in_channels, out_channels, bias=False)

        self.activation = torch.nn.LeakyReLU()

        self.attention = torch.nn.Parameter(
            torch.empty(1, n_heads, out_channels // n_heads)
        )

        self.basis_dim = basis_dim
        self.lin_E = torch.nn.Linear(basis_dim, out_channels, bias=False)

        self.output_layer = torch.nn.Sequential(
            torch.nn.LayerNorm(out_channels),
            activation,
        )

    def forward(
        self,
        senders: torch.Tensor, # real_hidden_channels
        receivers: torch.Tensor, # virt_hidden_channels (node)
        edge_indices: torch.Tensor, 
        edge_attrs: torch.Tensor,
        *args,
    ) -> torch.Tensor:
        """
        Forward pass for the attention block.

        Args:
            senders (torch.Tensor): Feature matrix of sender nodes (N_senders x in_channels).
            receivers (torch.Tensor): Feature matrix of receiver nodes (N_receivers x out_channels).
            edge_indices (torch.Tensor): Edge index tensor (2, n_edges).
            edge_attrs (torch.Tensor): Edge feature tensor (n_edges x basis_dim).

        Returns:
            torch.Tensor: Updated node embeddings (N_receivers x out_channels).
        """
        E = self.lin_E(edge_attrs)

        Q = self.lin_Q(receivers)[edge_indices[1]]
        K = self.lin_K(senders)[edge_indices[0]]
        V = self.lin_V(senders)[edge_indices[0]]

        weights = torch.sum(
            self.attention
            * self.activation((Q + K + E).view(-1, self.n_heads, self.hidden_channels)),
            dim=2,
        )
        weights = scatter_softmax(weights, edge_indices[1], dim=0)
        weights = weights.unsqueeze(-1)

        V = V.view(-1, self.n_heads, self.hidden_channels)

        embedding = scatter(
            (weights * V).view(-1, self.channels),
            edge_indices[1],
            reduce="add",
            dim=0,
            dim_size=receivers.shape[0],
        )

        embedding = self.output_layer(embedding)

        return embedding

    def reset_parameters(self):
        """
        Reinitializes the model parameters.
        """
        init_xavier_uniform(self.lin_E)
        init_xavier_uniform(self.lin_Q)
        init_xavier_uniform(self.lin_K)
        init_xavier_uniform(self.lin_V)
        glorot(self.attention)


class BroadcastBlock(torch.nn.Module):
    """
    Broadcast block that maps aggregated virtual-node embeddings back to node embeddings.

    Uses multi-head attention with learnable transforms for keys/values and an
    optional per-edge regularization weight.

    Initialization parameters
    -------------------------
    in_channels (int): dimensionality of node features used for broadcasting
    out_channels (int): dimensionality of the output node features
    activation (torch.nn.Module): activation used inside the output network
    n_heads (int): number of attention heads
    basis_dim (int): dimensionality of edge basis features
    """

    def __init__(
        self,
        in_channels: int,  # virt_hidden_channels (node)
        out_channels: int,  # real_hidden_channels
        n_heads: int,  # num_virt_heads
        basis_dim: int,  # virt_basis_dim (edge)
        activation: torch.nn.Module=torch.nn.SiLU(),
        **kwargs,
    ):
        super().__init__()

        if in_channels % n_heads != 0:
            raise ValueError(
                "The number of input attention channels must be divisible by the number of heads"
            )

        if out_channels % n_heads != 0:
            raise ValueError(
                "The number of output attention channels must be divisible by the number of heads"
            )

        self.n_heads = n_heads
        self.channels = in_channels
        self.hidden_channels = in_channels // self.n_heads

        self.lin_Q = torch.nn.Linear(out_channels, in_channels, bias=False)
        self.lin_K = torch.nn.Linear(out_channels, in_channels, bias=False)
        self.lin_V = torch.nn.Linear(out_channels, in_channels, bias=False)

        self.activation = torch.nn.LeakyReLU()

        self.attention = torch.nn.Parameter(
            torch.empty(1, n_heads, in_channels // n_heads)
        )

        self.basis_dim = basis_dim
        self.lin_E = torch.nn.Linear(basis_dim, in_channels, bias=False)

        self.weights_K = torch.nn.Parameter(
            torch.empty(n_heads, in_channels // n_heads, in_channels // n_heads)
        )
        self.weights_V = torch.nn.Parameter(
            torch.empty(n_heads, in_channels // n_heads, in_channels // n_heads)
        )

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels, bias=False),
            torch.nn.LayerNorm(in_channels),
            activation,
            torch.nn.Linear(in_channels, out_channels, bias=False),
        )

    def forward(
        self,
        senders: torch.Tensor,  # virt_hidden_channels (node)
        senders_self: torch.Tensor, 
        receivers: torch.Tensor,     # real_hidden_channels
        edge_indices: torch.Tensor,
        edge_attrs: torch.Tensor,
        regularization_weights: torch.Tensor,
        *args,
    ) -> torch.Tensor:
        """
        Forward pass for the attention block.

        Args:
            senders (torch.Tensor): Feature matrix of sender nodes (N_senders x in_channels).
            senders_self (torch.Tensor): Feature matrix of sender nodes in self-loops (N_receivers x in_channels).
            receivers (torch.Tensor): Feature matrix of receiver nodes (N_receivers x in_channels).
            edge_indices (torch.Tensor): Edge index tensor (2, n_edges).
            edge_attrs (torch.Tensor): Edge feature tensor (n_edges x basis_dim).
            regularization_weights (torch.Tensor): .

        Returns:
            torch.Tensor: Updated node embeddings.
        """
        K = torch.vmap(torch.matmul, in_dims=(1, 0), out_dims=1)(
            senders.view(-1, self.n_heads, self.hidden_channels), self.weights_K
        ).reshape(-1, self.channels)

        K_self = self.lin_K(senders_self)
        K = torch.cat([K, K_self], dim=0)[edge_indices[0]]

        V = torch.vmap(torch.matmul, in_dims=(1, 0), out_dims=1)(
            senders.view(-1, self.n_heads, self.hidden_channels), self.weights_V
        ).reshape(-1, self.channels)
        V_self = self.lin_V(senders_self)
        V = torch.cat([V, V_self], dim=0)[edge_indices[0]]

        E = self.lin_E(edge_attrs)
        Q = self.lin_Q(receivers)[edge_indices[1]]

        weights = torch.sum(
            self.attention
            * self.activation((Q + K + E).view(-1, self.n_heads, self.hidden_channels)),
            dim=2,
        )
        weights = scatter_softmax(weights, edge_indices[1], dim=0)

        weights = weights.unsqueeze(-1)

        V = V.view(-1, self.n_heads, self.hidden_channels)

        embedding = scatter(
            regularization_weights.unsqueeze(1) * (weights * V).view(-1, self.channels),
            edge_indices[1],
            reduce="add",
            dim=0,
            dim_size=receivers.shape[0],
        )

        embedding = self.output_layer(embedding)

        return embedding
    
    def reset_parameters(self):
        """
        Reinitializes the model parameters.
        """
        init_xavier_uniform(self.lin_E)
        init_xavier_uniform(self.lin_Q)
        init_xavier_uniform(self.lin_K)
        init_xavier_uniform(self.lin_V)
        glorot(self.attention)
        glorot(self.weights_K)
        glorot(self.weights_V)
        for layer in self.output_layer:
            init_xavier_uniform(layer)