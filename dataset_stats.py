"""Print the size and edge homophily of every dataset (Table 5.1 of the thesis)."""

import pandas as pd

from gnnbench.data import HETEROPHILOUS, HOMOPHILOUS, edge_homophily, load_dataset


def main():
    rows = []
    for name in HOMOPHILOUS + HETEROPHILOUS:
        dataset = load_dataset(name)
        data = dataset[0]
        rows.append(
            {
                "dataset": name,
                "nodes": data.num_nodes,
                "edges": data.num_edges // 2 if data.is_undirected() else data.num_edges,
                "features": dataset.num_node_features,
                "classes": dataset.num_classes,
                "homophily": round(edge_homophily(data), 2),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
