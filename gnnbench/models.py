"""Model definitions.

GCN, GAT, GraphSAGE, SGC, ChebNet and APPNP are thin wrappers around PyTorch
Geometric layers. H2GCN and the global Graph Transformer are implemented here.
Every model takes a PyG `Data` object and returns log-probabilities.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh
from torch.nn import Linear
from torch_geometric.nn import APPNP, ChebConv, GATConv, GCNConv, SAGEConv, SGConv
from torch_geometric.utils import to_scipy_sparse_matrix


class MLP(nn.Module):
    """Baseline that ignores the graph."""

    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x = F.relu(self.lin1(data.x))
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.lin2(x), dim=1)


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.conv2(x, data.edge_index), dim=1)


class GAT(nn.Module):
    """`hidden_channels` is per head: 8 heads of 8 channels give 64 in total."""

    def __init__(self, in_channels, out_channels, hidden_channels=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return F.log_softmax(self.conv2(x, data.edge_index), dim=1)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.conv2(x, data.edge_index), dim=1)


class SGC(nn.Module):
    """Linear model on pre-propagated features, so it has no hidden size."""

    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super().__init__()
        self.conv1 = SGConv(in_channels, out_channels, K=2, cached=True)

    def forward(self, data):
        return F.log_softmax(self.conv1(data.x, data.edge_index), dim=1)


class ChebNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=3, normalization="sym")
        self.conv2 = ChebConv(hidden_channels, out_channels, K=3, normalization="sym")

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.conv2(x, data.edge_index), dim=1)


class APPNPNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.prop1 = APPNP(K=10, alpha=0.1)

    def forward(self, data):
        x = F.dropout(data.x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.prop1(self.lin2(x), data.edge_index), dim=1)


def undirected_adjacency(edge_index, num_nodes):
    """Symmetric adjacency matrix without self-loops, as a scipy CSR matrix."""
    A = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
    A = A.maximum(A.T).tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def symmetric_normalize(A):
    """Return D^(-1/2) A D^(-1/2). Isolated nodes stay at zero."""
    degree = np.asarray(A.sum(axis=1)).flatten()
    inv_sqrt = np.zeros_like(degree, dtype=np.float64)
    inv_sqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    D_inv_sqrt = sp.diags(inv_sqrt, format="csr")
    return D_inv_sqrt @ A @ D_inv_sqrt


def to_torch_sparse(matrix, device):
    matrix = matrix.tocoo()
    indices = torch.tensor(np.vstack((matrix.row, matrix.col)), dtype=torch.long)
    values = torch.tensor(matrix.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=matrix.shape, device=device).coalesce()


class H2GCN(nn.Module):
    """H2GCN (Zhu et al., NeurIPS 2020).

    The three design choices that make it work under heterophily:
    1. ego and neighbor embeddings are kept apart (no self-loops);
    2. the 1-hop and strict 2-hop neighborhoods are aggregated separately;
    3. the classifier sees the representations of every round.

    Each round doubles the width, so with K=2 the output of the rounds is
    64, 128 and 256 channels and the classifier receives 448. Note that strict
    2-hop edges keep their path counts as weights, while the paper binarizes them.
    """

    def __init__(self, in_channels, out_channels, hidden_channels=64, K=2):
        super().__init__()
        self.K = K
        self.lin1 = Linear(in_channels, hidden_channels)
        self.classifier = Linear(hidden_channels * (2 ** (K + 1) - 1), out_channels)
        self._propagation = None

    @staticmethod
    def propagation_matrices(edge_index, num_nodes, device):
        A = undirected_adjacency(edge_index, num_nodes)

        # A^2 counts the paths of length 2. The strict 2-hop neighborhood drops
        # the node itself and everything already reachable in one hop.
        A2 = (A @ A).tolil()
        A2.setdiag(0)
        A2 = A2.tocsr()
        A2 = A2 - A.multiply(A2)
        A2.eliminate_zeros()

        return (
            to_torch_sparse(symmetric_normalize(A), device),
            to_torch_sparse(symmetric_normalize(A2), device),
        )

    def forward(self, data):
        if self._propagation is None:
            self._propagation = self.propagation_matrices(
                data.edge_index, data.num_nodes, data.x.device
            )
        R1, R2 = self._propagation

        h = F.relu(self.lin1(data.x))
        h = F.dropout(h, p=0.5, training=self.training)

        representations = [h]
        for _ in range(self.K):
            h = torch.cat([torch.sparse.mm(R1, h), torch.sparse.mm(R2, h)], dim=1)
            representations.append(h)

        return F.log_softmax(self.classifier(torch.cat(representations, dim=1)), dim=1)


def laplacian_positional_encoding(edge_index, num_nodes, k=8, seed=42):
    """Eigenvectors of the k smallest non-trivial Laplacian eigenvalues."""
    A = undirected_adjacency(edge_index, num_nodes)
    L = sp.eye(num_nodes, format="csr") - symmetric_normalize(A)

    # An eigenvector's sign is arbitrary and depends on the solver's start vector,
    # which is random and ignores np.random.seed unless it is set explicitly.
    v0 = np.random.default_rng(seed).normal(size=num_nodes)

    # One extra eigenvector, because the first one has eigenvalue 0.
    eigenvalues, eigenvectors = eigsh(L, k=k + 1, which="SM", v0=v0)
    pe = eigenvectors[:, np.argsort(eigenvalues)][:, 1:]
    return torch.tensor(pe, dtype=torch.float32)


class GlobalTransformerLayer(nn.Module):
    def __init__(self, hidden_channels=64, heads=4, dropout=0.5):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_channels, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            Linear(hidden_channels, hidden_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels * 4, hidden_channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attended))
        return self.norm2(x + self.dropout(self.ffn(x)))


class GraphTransformer(nn.Module):
    """Global self-attention over all nodes.

    Unlike message passing, attention is not limited to the neighbors: the graph
    only enters through the Laplacian positional encoding (`data.laplacian_pe`).
    The graph is treated as one sequence of N tokens, so memory grows as O(N^2).
    """

    def __init__(self, in_channels, out_channels, hidden_channels=64, pe_dim=8, heads=4, num_layers=2):
        super().__init__()
        self.input_projection = Linear(in_channels, hidden_channels)
        self.pe_projection = Linear(pe_dim, hidden_channels)
        self.layers = nn.ModuleList(
            GlobalTransformerLayer(hidden_channels, heads) for _ in range(num_layers)
        )
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x = self.input_projection(data.x) + self.pe_projection(data.laplacian_pe)
        x = F.dropout(x, p=0.5, training=self.training)

        x = x.unsqueeze(0)  # [N, D] -> [1, N, D]
        for layer in self.layers:
            x = layer(x)
        return F.log_softmax(self.classifier(x.squeeze(0)), dim=1)


MODELS = {
    "MLP": MLP,
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE,
    "SGC": SGC,
    "ChebNet": ChebNet,
    "APPNP": APPNPNet,
    "H2GCN": H2GCN,
    "GraphTransformer": GraphTransformer,
}
