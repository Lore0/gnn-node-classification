"""Seeding, metrics and the training loop shared by every experiment."""

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(out, y, mask, num_classes):
    y_true = y[mask].cpu().numpy()
    y_pred = out[mask].argmax(dim=1).cpu().numpy()
    return {
        "accuracy": float((y_pred == y_true).mean()) * 100.0,
        # Passing `labels` counts a class that is absent from the predictions as F1 = 0.
        "macro_f1": f1_score(
            y_true, y_pred, labels=np.arange(num_classes), average="macro", zero_division=0
        )
        * 100.0,
    }


def train_and_evaluate(model, data, num_classes, lr, weight_decay, max_epochs=1000, patience=100):
    """Train with early stopping on validation loss and score the best epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss, best_state, best_epoch = float("inf"), None, -1
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.nll_loss(model(data)[data.val_mask], data.y[data.val_mask]).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Validation loss was never finite.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data)

    return {
        "best_epoch": best_epoch + 1,
        "train": compute_metrics(out, data.y, data.train_mask, num_classes),
        "val": compute_metrics(out, data.y, data.val_mask, num_classes),
        "test": compute_metrics(out, data.y, data.test_mask, num_classes),
    }
