"""Run one of the two benchmarks and write the results to a CSV.

    python run_experiment.py homophilous
    python run_experiment.py heterophilous
    python run_experiment.py homophilous --no-normalize
    python run_experiment.py heterophilous --undirected
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected

from gnnbench.data import HETEROPHILOUS, HOMOPHILOUS, load_dataset, stratified_split
from gnnbench.models import MODELS, laplacian_positional_encoding
from gnnbench.training import set_seed, train_and_evaluate

EXPERIMENTS = {
    "homophilous": {
        "datasets": HOMOPHILOUS,
        "models": ["GCN", "GAT", "GraphSAGE", "SGC", "ChebNet", "APPNP"],
        "lr": 0.05,
        "weight_decay": 5e-4,
        # SGC is linear, so it gets its own learning rate and weight decay.
        "overrides": {"SGC": {"lr": 0.2, "weight_decay": 5e-5}},
    },
    "heterophilous": {
        "datasets": HETEROPHILOUS,
        "models": ["MLP", "GCN", "ChebNet", "H2GCN", "GraphTransformer"],
        "lr": 0.01,
        "weight_decay": 5e-4,
        "overrides": {},
    },
}


def prepare_data(dataset, normalize, undirected, seed, with_positional_encoding, device):
    data = dataset[0]
    if normalize:
        data = T.NormalizeFeatures()(data)
    if undirected:
        data.edge_index = to_undirected(data.edge_index)
    data = stratified_split(data, seed=seed)
    if with_positional_encoding:
        data.laplacian_pe = laplacian_positional_encoding(
            data.edge_index, data.num_nodes, seed=seed
        )
    return data.to(device)


def run(config, args):
    device = torch.device(args.device)
    rows = []

    for dataset_name in args.datasets or config["datasets"]:
        dataset = load_dataset(dataset_name)
        model_names = args.models or config["models"]
        data = prepare_data(
            dataset,
            not args.no_normalize,
            args.undirected,
            args.seed,
            "GraphTransformer" in model_names,
            device,
        )

        for model_name in model_names:
            set_seed(args.seed)
            model = MODELS[model_name](dataset.num_node_features, dataset.num_classes).to(device)
            params = config["overrides"].get(model_name, {})
            result = train_and_evaluate(
                model,
                data,
                dataset.num_classes,
                lr=params.get("lr", config["lr"]),
                weight_decay=params.get("weight_decay", config["weight_decay"]),
            )

            test = result["test"]
            print(
                f"{dataset_name:>10} {model_name:>16} | acc {test['accuracy']:6.2f} "
                f"| macro-F1 {test['macro_f1']:6.2f} | epoch {result['best_epoch']}",
                flush=True,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "best_epoch": result["best_epoch"],
                    "train_accuracy": round(result["train"]["accuracy"], 2),
                    "val_accuracy": round(result["val"]["accuracy"], 2),
                    "test_accuracy": round(test["accuracy"], 2),
                    "test_macro_f1": round(test["macro_f1"], 2),
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--no-normalize", action="store_true", help="keep the raw features")
    parser.add_argument("--undirected", action="store_true", help="symmetrize the edges")
    parser.add_argument("--datasets", nargs="+", help="subset of the experiment's datasets")
    parser.add_argument("--models", nargs="+", choices=MODELS, help="subset of the models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="default: results/<experiment>[_raw][_undirected].csv")
    args = parser.parse_args()

    suffix = ("_raw" if args.no_normalize else "") + ("_undirected" if args.undirected else "")
    output = args.output or Path("results") / f"{args.experiment}{suffix}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    results = run(EXPERIMENTS[args.experiment], args)
    results.to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
