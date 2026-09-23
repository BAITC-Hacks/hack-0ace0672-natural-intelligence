#!/usr/bin/env python3
"""Пайплайн «Граф денег»: три parquet -> три CSV + graph.json.

    python run.py

Работает офлайн, без ключей, без GPU. Владелец файла: kuanyshs.
"""

import argparse
import time
from pathlib import Path

from src import clusters, features, priority, roles, schema
from src.io import DATA_DIR, OUT_DIR, build_graph, load, seeds
from src.mock import write_graph_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()

    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)

    edges, nodes, tx = load(a.data)
    G = build_graph(edges)
    seed_list = seeds(nodes)
    print(f"граф: {G.number_of_nodes()} узлов в рёбрах, {G.number_of_edges()} рёбер, {len(seed_list)} seed")

    df = features.compute(G, nodes, tx, seed_list)
    print(f"признаки посчитаны: {df.shape[1]} колонок")

    df = roles.assign(df, G)
    df = clusters.assign(df, G)
    df = priority.score(df)

    print("\nраспределение ролей:")
    print(df.role.value_counts().to_string())
    print("\n" + priority.sanity(df))

    extra = ["depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt",
             "transit_ratio", "net_flow", "external_funding_gap", "taint_share",
             "external_share", "ppr", "n_seeds_upstream", "median_delay_days",
             "hhi_in", "hhi_out", "in_cycle_le6", "was_expanded"]
    df[schema.NODES_ROLES_COLUMNS + extra].to_csv(a.out / "nodes_roles.csv", index=False)
    clusters.summarize(df, edges).to_csv(a.out / "clusters.csv", index=False)
    priority.top_nodes(df).to_csv(a.out / "top_nodes.csv", index=False)
    write_graph_json(df, edges, a.out / "graph.json")

    print()
    schema.validate(a.out)
    print(f"готово за {time.time() - t0:.1f} с -> {a.out}/")


if __name__ == "__main__":
    main()
