"""Фиктивные выгрузки в правильной схеме — чтобы фронт и агент стартовали немедленно.

    python -m src.mock

Роли и скоры здесь случайные, но структура файлов и graph.json точно такая же,
как у настоящего пайплайна. Когда kuanyshs выдаст реальные выгрузки,
переключение будет бесшовным: те же файлы, те же колонки.
"""

import json

import numpy as np
import pandas as pd

from src import schema
from src.io import DATA_DIR, OUT_DIR, build_graph, load

RNG = np.random.default_rng(42)


def main() -> None:
    edges, nodes, _ = load(DATA_DIR)
    G = build_graph(edges)
    OUT_DIR.mkdir(exist_ok=True)

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    in_kzt = dict(G.in_degree(weight="sum_kzt"))
    out_kzt = dict(G.out_degree(weight="sum_kzt"))

    df = nodes[["gid", "depth", "is_seed"]].copy()
    df["in_deg"] = df.gid.map(in_deg).fillna(0).astype(int)
    df["out_deg"] = df.gid.map(out_deg).fillna(0).astype(int)
    df["in_kzt"] = df.gid.map(in_kzt).fillna(0.0)
    df["out_kzt"] = df.gid.map(out_kzt).fillna(0.0)

    # Грубая раскладка ролей — только чтобы на экране были все цвета.
    role = np.where(
        df.out_deg == 0,
        np.where(df.depth == 4, "terminal_unknown", "terminal"),
        np.where(df.out_deg >= 10, "distributor",
                 np.where(df.in_deg >= 5, "consolidator",
                          np.where(df.in_deg + df.out_deg <= 2, "peripheral", "transit"))),
    )
    df["role"] = role
    # немного coordinator, чтобы роль не была пустой на экране
    cand = df.index[(df.out_deg >= 5) & (df.in_deg >= 3)]
    df.loc[RNG.choice(cand, size=min(30, len(cand)), replace=False), "role"] = "coordinator"

    df["role_score"] = RNG.uniform(0.3, 0.95, len(df)).round(3)
    df["priority_score"] = RNG.uniform(0, 1, len(df)).round(3)
    df["cluster_id"] = RNG.integers(0, 8, len(df))
    df["evidence"] = [
        f"MOCK: in_deg={a}, out_deg={b}, in={c:,.0f} KZT, out={d:,.0f} KZT"
        for a, b, c, d in zip(df.in_deg, df.out_deg, df.in_kzt, df.out_kzt)
    ]

    df[schema.NODES_ROLES_COLUMNS + ["depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt"]] \
        .to_csv(OUT_DIR / "nodes_roles.csv", index=False)

    clusters = (
        df.groupby("cluster_id")
        .agg(n_nodes=("gid", "size"), n_seed=("is_seed", "sum"), sum_kzt_internal=("out_kzt", "sum"))
        .reset_index()
    )
    clusters["top_gids"] = [
        ";".join(map(str, df[df.cluster_id == c].nlargest(3, "priority_score").gid))
        for c in clusters.cluster_id
    ]
    clusters["hypothesis"] = "MOCK: гипотеза будет сгенерирована по составу ролей"
    clusters[schema.CLUSTERS_COLUMNS].to_csv(OUT_DIR / "clusters.csv", index=False)

    top = df.nlargest(25, "priority_score").reset_index(drop=True)
    top["rank"] = top.index + 1
    top["why"] = "MOCK: обоснование появится вместе с настоящим priority_score"
    top[schema.TOP_NODES_COLUMNS].to_csv(OUT_DIR / "top_nodes.csv", index=False)

    write_graph_json(df, edges, OUT_DIR / "graph.json")
    print(f"Mock-выгрузки записаны в {OUT_DIR}/ — роли и скоры случайные")


def write_graph_json(df: pd.DataFrame, edges: pd.DataFrame, path,
                     top: int = 120, max_nodes: int = 500) -> None:
    """Топ-N по приоритету плюс их соседи, с жёстким потолком по числу узлов.

    Потолок обязателен: на 1800 узлах раскладка вешает вкладку.
    Остальные узлы доступны через поиск по gid.
    Формат потребляет web/index.html. Реальный пайплайн пишет его же.
    """
    core = set(df.nlargest(top, "priority_score").gid)
    nbr = edges[edges.src.isin(core) | edges.dst.isin(core)]
    extra = (set(nbr.src) | set(nbr.dst)) - core
    # соседей добираем по приоритету, а не как попало
    ranked = df[df.gid.isin(extra)].nlargest(max(0, max_nodes - len(core)), "priority_score").gid
    keep = core | set(ranked)
    nbr = edges[edges.src.isin(keep) & edges.dst.isin(keep)]

    sub = df[df.gid.isin(keep)]
    payload = {
        "nodes": [
            {
                "id": str(r.gid),
                "role": r.role,
                "priority": float(r.priority_score),
                "cluster": int(r.cluster_id),
                "depth": int(r.depth),
                "is_seed": bool(r.is_seed),
            }
            for r in sub.itertuples(index=False)
        ],
        "edges": [
            {"source": str(r.src), "target": str(r.dst), "sum_kzt": float(r.sum_kzt), "n_tx": int(r.n_tx)}
            for r in nbr.itertuples(index=False)
            if r.src in keep and r.dst in keep
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
