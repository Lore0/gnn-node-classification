"""Draw the accuracy and macro-F1 bar charts from a results CSV.

    python plot_results.py results/thesis_homophilous.csv figures/homophilous.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#1D3557", "#E63946", "#2A9D8F", "#F4A261", "#8AB17D", "#9D4EDD"]


def plot(results, output):
    datasets = results["dataset"].unique()
    models = results["model"].unique()
    width = 0.8 / len(models)
    x = np.arange(len(datasets))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    panels = [("test_accuracy", "Accuracy (%)"), ("test_macro_f1", "Macro-F1 (%)")]

    for ax, (metric, label) in zip(axes, panels):
        table = results.pivot(index="dataset", columns="model", values=metric)
        for i, model in enumerate(models):
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(x + offset, table.loc[datasets, model], width, label=model, color=COLORS[i], zorder=3)
        ax.set_ylabel(label)
        ax.set_xticks(x, datasets)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(pd.read_csv(args.results), args.output)


if __name__ == "__main__":
    main()
