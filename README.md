# GNN Node Classification Benchmark

Code for the experimental chapter of my bachelor's thesis, *Node Classification: Formulation, Methodologies and Applications* (University of Padua, 2025/26, supervisor Prof. Ornella Irrera).

It compares 8 graph models and an MLP baseline on 11 datasets, to answer one question: when does using the graph help, and when does it hurt? H2GCN and the Graph Transformer are implemented from scratch in PyTorch. The other models use PyTorch Geometric layers.

## Key findings

- **Message passing can lose to a model that ignores the graph.** On Wisconsin (homophily 0.20) an MLP reaches 76.6% accuracy against 49.8% for GCN.
- **Preprocessing can decide the ranking.** Row-normalizing the features took SGC from 92.8% to 44.3% on Amazon Photo, but without it GraphSAGE fell from 91.3% to 47.5%.
- **Edge direction matters as much as the architecture.** H2GCN looked best on Chameleon because it symmetrizes the edges and GCN did not. With undirected edges GCN goes from 33.1% to 49.8% and beats it.

The last two came from checks I ran after the thesis. See [the analysis notes](docs/analysis.md) for the details, the protocol and the limitations.

## Results

Numbers from the thesis: test accuracy / macro-F1 in %, with the best accuracy per dataset in bold. They come from a single seed and split, so gaps of a few points mean little.

### Homophilous graphs

| Dataset | Homophily | GCN | GAT | GraphSAGE | SGC | ChebNet | APPNP |
|---|---|---|---|---|---|---|---|
| Cora | 0.81 | 83.87 / 82.61 | 82.67 / 81.48 | 81.89 / 80.05 | 76.99 / 73.25 | 81.38 / 80.42 | **85.40** / 84.22 |
| CiteSeer | 0.74 | 72.60 / 67.53 | 71.85 / 64.70 | 71.93 / 67.20 | **73.73** / 64.43 | 71.55 / 66.27 | 73.09 / 67.34 |
| PubMed | 0.80 | **86.31** / 85.67 | 84.85 / 83.99 | 86.04 / 85.70 | 83.63 / 82.95 | 86.26 / 85.94 | 85.77 / 85.07 |
| DBLP | 0.83 | 83.52 / 78.88 | 82.91 / 77.04 | 81.88 / 77.24 | 80.56 / 72.40 | 82.13 / 75.84 | **84.37** / 79.84 |
| Cora-ML | 0.79 | 84.61 / 83.03 | 82.31 / 78.21 | 83.31 / 82.10 | 61.54 / 52.81 | 83.52 / 82.02 | **84.98** / 82.87 |
| Coauthor CS | 0.81 | 89.03 / 74.83 | 87.80 / 71.82 | **93.13** / 90.29 | 75.02 / 50.17 | 92.96 / 90.21 | 87.45 / 69.73 |
| Amazon Photo | 0.83 | 84.44 / 74.49 | 80.88 / 70.58 | **91.50** / 85.53 | 44.30 / 16.47 | 89.95 / 83.58 | 83.01 / 73.71 |

![Accuracy and macro-F1 on the homophilous datasets](figures/homophilous.png)

### Heterophilous graphs

| Dataset | Homophily | MLP | GCN | ChebNet | H2GCN | Graph Transformer |
|---|---|---|---|---|---|---|
| Texas | 0.11 | 55.78 / 17.87 | 55.10 / 14.27 | 56.46 / 21.39 | **70.75** / 45.72 | 55.10 / 14.21 |
| Wisconsin | 0.20 | **76.62** / 46.27 | 49.75 / 18.87 | 64.18 / 37.53 | 61.19 / 31.87 | 46.77 / 12.75 |
| Actor | 0.22 | **34.51** / 29.55 | 27.43 / 18.32 | 31.86 / 26.98 | 33.82 / 28.90 | 25.87 / 8.22 |
| Chameleon | 0.24 | 32.07 / 27.85 | 30.64 / 28.26 | 34.38 / 33.66 | **45.03** / 44.58 | 36.41 / 34.34 |

![Accuracy and macro-F1 on the heterophilous datasets](figures/heterophilous.png)

## Usage

```bash
pip install -r requirements.txt

python run_experiment.py homophilous
python run_experiment.py heterophilous --datasets Texas --models H2GCN
python run_experiment.py homophilous --no-normalize
python run_experiment.py heterophilous --undirected

python dataset_stats.py      # size and edge homophily of every dataset
python grid_search.py        # hidden size / learning rate search on the homophilous datasets
python plot_results.py results/thesis_homophilous.csv figures/homophilous.png
```

Datasets are downloaded to `data/` on first use and results are written to `results/`. A GPU is strongly recommended for the homophilous experiment.

## License

MIT
