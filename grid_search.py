"""Grid search over hidden size and learning rate on the homophilous datasets.

Each configuration is averaged over three random splits, with a shorter training
budget than the final runs. Validation and test scores are both logged.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T

from gnnbench.data import HOMOPHILOUS, load_dataset, stratified_split
from gnnbench.models import MODELS
from gnnbench.training import train_and_evaluate

MODEL_NAMES = ["GCN", "GAT", "GraphSAGE", "SGC", "ChebNet", "APPNP"]
HIDDEN_SIZES = [16, 64]
LEARNING_RATES = [0.01, 0.05]
SGC_LEARNING_RATES = [0.05, 0.2]
SEEDS = [42, 123, 999]


def evaluate_configuration(ModelClass, dataset, data, hidden, lr, weight_decay, device):
    val_scores, test_scores, f1_scores = [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        split = stratified_split(data.clone(), seed=seed).to(device)
        model = ModelClass(dataset.num_node_features, dataset.num_classes, hidden).to(device)
        result = train_and_evaluate(
            model, split, dataset.num_classes, lr, weight_decay, max_epochs=300, patience=30
        )
        val_scores.append(result["val"]["accuracy"])
        test_scores.append(result["test"]["accuracy"])
        f1_scores.append(result["test"]["macro_f1"])
    return val_scores, test_scores, f1_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/grid_search.csv"))
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_name in HOMOPHILOUS:
        dataset = load_dataset(dataset_name)
        data = T.NormalizeFeatures()(dataset[0])

        for model_name in MODEL_NAMES:
            is_sgc = model_name == "SGC"
            hidden_sizes = [None] if is_sgc else HIDDEN_SIZES
            learning_rates = SGC_LEARNING_RATES if is_sgc else LEARNING_RATES
            weight_decay = 5e-5 if is_sgc else 5e-4

            for hidden, lr in itertools.product(hidden_sizes, learning_rates):
                val, test, f1 = evaluate_configuration(
                    MODELS[model_name], dataset, data, hidden, lr, weight_decay, device
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "hidden": hidden,
                        "lr": lr,
                        "val_accuracy_mean": round(np.mean(val), 2),
                        "test_accuracy_mean": round(np.mean(test), 2),
                        "test_accuracy_std": round(np.std(test), 2),
                        "test_macro_f1_mean": round(np.mean(f1), 2),
                    }
                )
                print(
                    f"{dataset_name:>10} {model_name:>10} hidden={hidden} lr={lr} "
                    f"| val {np.mean(val):.2f} | test {np.mean(test):.2f} +- {np.std(test):.2f}",
                    flush=True,
                )
                # Saved after every configuration, so a crash keeps what is done.
                pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
