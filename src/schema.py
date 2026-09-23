"""Замороженный контракт выгрузок. Меняется только по общему согласию команды.

Схема зафиксирована в ТЗ, жюри проверяет её механически.
Лишние колонки добавлять можно, обязательные удалять — нет.
"""

from pathlib import Path

import pandas as pd

N_NODES = 2248
N_EDGES = 3119
N_TX = 4840
N_SEED = 81

# Словарь ролей из ТЗ + одно документированное расширение.
# terminal_unknown отделяет узлы, обрезанные 4-м коленом обхода, от настоящих стоков.
ROLES = [
    "consolidator",
    "transit",
    "distributor",
    "terminal",
    "terminal_unknown",
    "coordinator",
    "peripheral",
]

NODES_ROLES_COLUMNS = [
    "gid",
    "role",
    "role_score",
    "cluster_id",
    "priority_score",
    "evidence",
]

CLUSTERS_COLUMNS = [
    "cluster_id",
    "n_nodes",
    "n_seed",
    "sum_kzt_internal",
    "top_gids",
    "hypothesis",
]

TOP_NODES_COLUMNS = ["rank", "gid", "role", "priority_score", "why"]

EVIDENCE_MAX_LEN = 200


def validate(out_dir: Path) -> None:
    """Проверяет выгрузки ровно так, как это будет делать жюри.

    Падает с внятным сообщением — лучше упасть у себя, чем на демо.
    """
    nodes = pd.read_csv(out_dir / "nodes_roles.csv")
    clusters = pd.read_csv(out_dir / "clusters.csv")
    top = pd.read_csv(out_dir / "top_nodes.csv")

    problems = []

    missing = [c for c in NODES_ROLES_COLUMNS if c not in nodes.columns]
    if missing:
        problems.append(f"nodes_roles.csv: нет колонок {missing}")

    if len(nodes) != N_NODES:
        problems.append(f"nodes_roles.csv: {len(nodes)} строк вместо {N_NODES}")

    if "gid" in nodes and nodes.gid.duplicated().any():
        problems.append("nodes_roles.csv: есть дубли gid")

    if "role" in nodes:
        bad = sorted(set(nodes.role.dropna()) - set(ROLES))
        if bad:
            problems.append(f"nodes_roles.csv: роли вне словаря: {bad}")
        if nodes.role.isna().any() or (nodes.role == "").any():
            problems.append("nodes_roles.csv: есть узлы без роли")

    if "evidence" in nodes:
        empty = nodes.evidence.isna() | (nodes.evidence.astype(str).str.strip() == "")
        if empty.any():
            problems.append(f"nodes_roles.csv: пустой evidence у {int(empty.sum())} узлов")
        too_long = nodes.evidence.astype(str).str.len() > EVIDENCE_MAX_LEN
        if too_long.any():
            problems.append(f"nodes_roles.csv: evidence >200 символов у {int(too_long.sum())} узлов")

    for col in ("role_score", "priority_score"):
        if col in nodes:
            out_of_range = ~nodes[col].between(0, 1)
            if out_of_range.any():
                problems.append(f"nodes_roles.csv: {col} вне [0,1] у {int(out_of_range.sum())} узлов")

    if "cluster_id" in nodes and (nodes.cluster_id.isna() | (nodes.cluster_id < 0)).any():
        problems.append("nodes_roles.csv: есть узлы без cluster_id")

    missing = [c for c in CLUSTERS_COLUMNS if c not in clusters.columns]
    if missing:
        problems.append(f"clusters.csv: нет колонок {missing}")
    if len(clusters) == 0:
        problems.append("clusters.csv: пустой")

    missing = [c for c in TOP_NODES_COLUMNS if c not in top.columns]
    if missing:
        problems.append(f"top_nodes.csv: нет колонок {missing}")
    if len(top) < 20:
        problems.append(f"top_nodes.csv: {len(top)} строк, нужно >=20")

    if problems:
        raise SystemExit("ВЫГРУЗКИ НЕ ПРОШЛИ ПРОВЕРКУ:\n  - " + "\n  - ".join(problems))

    print(f"Проверка выгрузок пройдена: {len(nodes)} узлов, "
          f"{len(clusters)} кластеров, {len(top)} строк в топе")
