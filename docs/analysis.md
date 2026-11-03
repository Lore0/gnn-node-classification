# Analysis notes

Details behind the [README](../README.md): the checks I ran after the thesis, the protocol, reproducibility and limitations.

## Feature normalization changes the ranking

The thesis applied `NormalizeFeatures` everywhere. Rerunning every model with and without it gives this test accuracy (`results/normalization_check.csv`):

| Model | CS, normalized | CS, raw | Photo, normalized | Photo, raw |
|---|---|---|---|---|
| GCN | 89.22 | 92.88 | 84.49 | 92.70 |
| GAT | 87.75 | 92.49 | 87.48 | **93.48** |
| APPNP | 87.57 | **93.16** | 83.61 | 92.12 |
| SGC | 75.02 | 93.09 | 44.30 | 92.83 |
| GraphSAGE | 93.13 | 92.84 | 91.32 | 47.45 |
| ChebNet | 92.98 | out of memory | 90.08 | 41.88 |

- **Why SGC collapses:** normalization divides each row by its sum, so Photo's feature norm drops from about 15 to 0.08. SGC is linear, so it would need huge weights, while weight decay pulls them toward zero. The logits stay almost equal, the bias decides, and the model predicts the majority class.
- **Why GraphSAGE and ChebNet collapse on raw Photo:** this is the opposite problem, large inputs at a learning rate of 0.05 make training unstable. My guess is that their non-aggregated "self" term passes the raw input straight through, but I haven't verified it.
- **What it means:** GraphSAGE's lead on CS and Photo mostly comes from this preprocessing choice. Without normalization GCN, GAT, APPNP and SGC all reach about 93%. Preprocessing should be selected on validation data for each model, like any other hyperparameter.

To reproduce: `python run_experiment.py homophilous` for the normalized runs and `python run_experiment.py homophilous --no-normalize` for the raw ones, which writes `results/homophilous_raw.csv`. `normalization_check.csv` puts both in one file and was produced by an earlier version of the script, so it has a `normalized` column and no `val_accuracy`.

## Edge direction changes the result on Chameleon

The four heterophilous graphs are directed in PyTorch Geometric. GCN and ChebNet used the edges as they are, while H2GCN and the Graph Transformer symmetrize them internally, so the comparison wasn't even. This is the test accuracy / macro-F1 with the same CPU setup (directed runs in `results/heterophilous_cpu.csv`, undirected runs in `results/heterophilous_undirected.csv`):

| Dataset | GCN, directed | GCN, undirected | ChebNet, directed | ChebNet, undirected | H2GCN |
|---|---|---|---|---|---|
| Texas | 51.70 / 14.91 | 56.46 / 20.74 | 55.78 / 20.01 | 60.54 / 29.10 | **70.07** / 44.46 |
| Wisconsin | 49.25 / 17.17 | 56.22 / 29.36 | 63.18 / 36.68 | **66.67** / 36.06 | 63.68 / 33.66 |
| Actor | 26.99 / 18.10 | 29.31 / 21.81 | 31.05 / 24.55 | 32.35 / 26.89 | **33.75** / 28.25 |
| Chameleon | 33.06 / 31.21 | **49.81** / 49.86 | 33.22 / 32.29 | 37.34 / 35.91 | 44.92 / 44.82 |

On Chameleon H2GCN loses its lead: plain GCN wins once it gets the same undirected graph. On Texas it stays ahead here, but that lead depends on the seed: with seeds 1, 2 and 3 H2GCN reaches 64.63, 59.18 and 78.23% accuracy, against 65.31, 61.90 and 68.03% for the MLP.

To reproduce: `python run_experiment.py heterophilous --device cpu --output results/heterophilous_cpu.csv` for the directed runs and `python run_experiment.py heterophilous --undirected --models GCN ChebNet` for the undirected ones.

## Protocol

- **Split:** stratified 10% train / 10% validation / 80% test.
- **Features:** `NormalizeFeatures`.
- **Training:** Adam and NLL loss, up to 1000 epochs, early stopping on validation loss with patience 100, restoring the best epoch.
- **Homophilous experiment:** learning rate 0.05 and weight decay 5e-4. SGC uses 0.2 and 5e-5.
- **Heterophilous experiment:** learning rate 0.01 and weight decay 5e-4.
- **Architectures:** 64 hidden channels, dropout 0.5. GAT uses 8 heads of 8 channels and dropout 0.6. ChebNet has K=3, SGC K=2, APPNP K=10 and alpha=0.1, H2GCN K=2. The Graph Transformer has 2 layers, 4 heads and an 8-dimensional positional encoding.
- **Metrics:** accuracy and macro-F1 on the test set.

## Reproducibility

This code is a refactor of the original Colab notebook. I checked it by running both versions on the same CPU setup, comparing accuracy, macro-F1 and best epoch:

- **Heterophilous experiment:** 18 of 20 runs are identical. The other two are Graph Transformer runs, explained below.
- **Homophilous experiment:** all 18 runs on Cora, CiteSeer and Cora-ML are identical, covering all six models. PubMed, DBLP, Coauthor CS and Amazon Photo were not compared.

The numbers won't match the thesis tables exactly, because those came from a Colab GPU. The CPU runs here differ by up to 6.4 points with the same seed and split (MLP on Chameleon: 32.07 on GPU, 38.50 on CPU).

With SciPy 1.17 the original Graph Transformer code isn't deterministic: the eigensolver starts from a random vector that `np.random.seed` doesn't control, so the signs of the positional encoding change on every run. This code derives the start vector from the seed, and repeated runs are now identical. The sign choice alone moves Chameleon between 31.80% and 38.44%.

## Limitations

- **One seed and one split.** Gaps of a few points mean little. GAT on Photo scored 80.88% in the thesis and 87.48% when the same code was rerun on GPU.
- **Hyperparameters weren't chosen on validation data.** The notebook's grid search logged test scores only. `grid_search.py` now logs validation scores too, but the thesis values are unchanged.
- **The split isn't the standard one.** 10/10/80 can't be compared with published results, which use the Planetoid or Geom-GCN splits.
- **Texas and Wisconsin are very small** (183 and 251 nodes), so a few nodes move accuracy by whole points.
- **Texas has a single-node class.** That node always lands in the test set and can't be learned, so macro-F1 there can't exceed 80%.
- **H2GCN differs slightly from the paper.** Strict 2-hop edges keep their path counts as weights, while the paper binarizes them.
- **Heterophilous models weren't tuned individually.** They all share one learning rate and weight decay.
