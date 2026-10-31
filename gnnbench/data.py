"""Dataset loading, the stratified split, and graph statistics."""

import torch
from torch_geometric.datasets import (
    Actor,
    Amazon,
    CitationFull,
    Coauthor,
    Planetoid,
    WebKB,
    WikipediaNetwork,
)

HOMOPHILOUS = ["Cora", "CiteSeer", "PubMed", "DBLP", "Cora_ML", "CS", "Photo"]
HETEROPHILOUS = ["Texas", "Wisconsin", "Actor", "Chameleon"]


def load_dataset(name, root="data"):
    if name in ("Cora", "CiteSeer", "PubMed"):
        return Planetoid(root=f"{root}/Planetoid", name=name)
    if name in ("DBLP", "Cora_ML"):
        return CitationFull(root=f"{root}/CitationFull", name=name)
    if name == "CS":
        return Coauthor(root=f"{root}/Coauthor", name=name)
    if name == "Photo":
        return Amazon(root=f"{root}/Amazon", name=name)
    if name in ("Texas", "Wisconsin"):
        return WebKB(root=f"{root}/WebKB", name=name)
    if name == "Actor":
        return Actor(root=f"{root}/Actor")
    if name == "Chameleon":
        return WikipediaNetwork(root=f"{root}/WikipediaNetwork", name=name)
    raise ValueError(f"Unknown dataset: {name}")


def stratified_split(data, train_ratio=0.10, val_ratio=0.10, seed=42):
    """Add train/val/test masks, split class by class.

    Every node left out of train and val goes to test. That includes all the
    nodes of a class with fewer than three members, which can't be represented
    in all three sets (Texas has a single-node class).
    """
    y = data.y.detach().cpu()
    train_mask = torch.zeros(y.numel(), dtype=torch.bool)
    val_mask = torch.zeros(y.numel(), dtype=torch.bool)
    generator = torch.Generator().manual_seed(seed)

    for class_id in torch.unique(y, sorted=True):
        class_indices = (y == class_id).nonzero(as_tuple=False).view(-1)
        class_size = class_indices.numel()
        if class_size < 3:
            continue

        class_indices = class_indices[torch.randperm(class_size, generator=generator)]
        n_train = max(1, round(train_ratio * class_size))
        n_val = max(1, round(val_ratio * class_size))
        if n_train + n_val >= class_size:
            n_val = 1
            n_train = class_size - 2

        train_mask[class_indices[:n_train]] = True
        val_mask[class_indices[n_train : n_train + n_val]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = ~(train_mask | val_mask)
    return data


def edge_homophily(data):
    """Fraction of edges that connect two nodes of the same class."""
    row, col = data.edge_index
    return (data.y[row] == data.y[col]).float().mean().item()
