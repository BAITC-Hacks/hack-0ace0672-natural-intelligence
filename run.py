#!/usr/bin/env python3
"""Пайплайн «Граф денег»: три parquet -> три CSV + graph.json.

    python run.py

Работает офлайн, без ключей, без GPU. Владелец файла: kuanyshs.
"""

import argparse
import time
from pathlib import Path

from src import clusters, features, motifs, priority, resilience, roles, schema, text
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

    # Аддитивный слой: четыре колонки и дописка в evidence. Роли и пороги не трогает.
    df, motif_pairs = motifs.compute(G, tx, df)
    df = motifs.augment_evidence(df)
    print(f"мотивы: {text.nodes(int(df.cycle_time_ok.sum()))} в циклах с верной хронологией, "
          f"{text.nodes(int((df.passthrough_matches > 0).sum()))} со сквозными платежами, "
          f"{len(motif_pairs)} пар в motifs.csv")

    df = clusters.assign(df, G)
    df = priority.score(df)

    print("\nраспределение ролей:")
    print(df.role.value_counts().to_string())
    print("\n" + priority.sanity(df))

    res = resilience.report(df, G)
    print("\nУСТОЙЧИВОСТЬ СЕТИ при изъятии топ-N узлов (опциональный пункт ТЗ)")
    print(resilience.render(res))
    res.to_csv(a.out / "resilience.csv", index=False)

    extra = ["depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt",
             "transit_ratio", "net_flow", "external_funding_gap", "taint_share",
             "external_share", "ppr", "n_seeds_upstream", "median_delay_days",
             "hhi_in", "hhi_out", "in_cycle_le6", "was_expanded",
             "in_tx", "out_tx", "max_edge_n_tx", "sync_in_events", "sent_before_received",
             "cycle_time_ok", "passthrough_matches",
             "shared_receivers_max", "shared_receivers_z"]
    df[schema.NODES_ROLES_COLUMNS + extra].to_csv(a.out / "nodes_roles.csv", index=False)
    clusters.summarize(df, edges).to_csv(a.out / "clusters.csv", index=False)
    priority.top_nodes(df).to_csv(a.out / "top_nodes.csv", index=False)
    motif_pairs.to_csv(a.out / "motifs.csv", index=False)
    write_graph_json(df, edges, a.out / "graph.json")

    # Автономная схема: если app.py не поднимется на демо, must-have №5
    # всё равно закрыт файлом, который открывается двойным кликом.
    try:
        from src import viz
        built = viz.build(a.out)
        # pyvis на Windows пишет файл в кодировке по умолчанию и молча оставляет
        # 0 байт, если в подписях есть кириллица. Пустая страховка хуже отсутствующей.
        if built.stat().st_size < 10_000:
            print(f"ВНИМАНИЕ: {built.name} собран пустым ({built.stat().st_size} байт) — "
                  f"автономная схема не откроется")
    except Exception as exc:
        print(f"graph.html не собран ({exc}) — интерфейс через app.py остаётся")

    print()
    schema.validate(a.out)
    print(f"готово за {time.time() - t0:.1f} с -> {a.out}/")


if __name__ == "__main__":
    main()
