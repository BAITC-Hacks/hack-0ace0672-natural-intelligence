"""Инструменты для ассистента: обычные функции на pandas. Владелец: narekdavtyan228-svg.

Они же используются шаблонным фолбэком, когда LLM недоступен.
Каждая функция возвращает dict, пригодный и для JSON, и для подстановки в шаблон.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "out"


@lru_cache(maxsize=1)
def _data():
    nodes = pd.read_csv(OUT / "nodes_roles.csv")
    clusters = pd.read_csv(OUT / "clusters.csv")
    top = pd.read_csv(OUT / "top_nodes.csv")
    edges = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "edges.parquet")
    return nodes, clusters, top, edges


def get_node(gid: int) -> dict:
    """Карточка узла: роль, уверенность, обоснование, ключевые числа."""
    nodes, _, _, edges = _data()
    row = nodes[nodes.gid == gid]
    if row.empty:
        return {"error": f"gid {gid} не найден"}
    r = row.iloc[0].to_dict()
    senders = edges[edges.dst == gid].nlargest(5, "sum_kzt")[["src", "sum_kzt", "n_tx"]]
    receivers = edges[edges.src == gid].nlargest(5, "sum_kzt")[["dst", "sum_kzt", "n_tx"]]
    r["top_senders"] = senders.to_dict("records")
    r["top_receivers"] = receivers.to_dict("records")
    return r


def who_collects_from(gids: list[int], max_depth: int = 3) -> dict:
    """Общие получатели ниже по течению от указанных узлов.

    Прямой ответ на сценарий ТЗ: «кто собирает деньги с этих пятерых?».
    """
    nodes, _, _, edges = _data()
    reach: dict[int, set[int]] = {}
    for g in gids:
        seen, frontier = set(), {g}
        for _ in range(max_depth):
            frontier = set(edges[edges.src.isin(frontier)].dst) - seen
            if not frontier:
                break
            seen |= frontier
        reach[g] = seen

    common: dict[int, int] = {}
    for s in reach.values():
        for n in s:
            common[n] = common.get(n, 0) + 1

    res = (
        nodes[nodes.gid.isin(common)]
        .assign(reached_from=lambda d: d.gid.map(common))
        .sort_values(["reached_from", "priority_score"], ascending=False)
        .head(10)
    )
    return {
        "asked_gids": gids,
        "collectors": res[["gid", "role", "priority_score", "reached_from", "evidence"]].to_dict("records"),
    }


def trace_path(src: int, dst: int, max_len: int = 6) -> dict:
    """Путь движения денег между двумя узлами."""
    import networkx as nx
    _, _, _, edges = _data()
    G = nx.DiGraph()
    for r in edges.itertuples(index=False):
        G.add_edge(r.src, r.dst, sum_kzt=float(r.sum_kzt))
    if src not in G or dst not in G:
        return {"error": "один из gid отсутствует в графе"}
    try:
        paths = list(nx.all_simple_paths(G, src, dst, cutoff=max_len))[:5]
    except nx.NetworkXNoPath:
        paths = []
    return {"src": src, "dst": dst, "paths": paths, "found": len(paths)}


def top_nodes(role: str | None = None, cluster_id: int | None = None, n: int = 10) -> dict:
    """Топ по приоритету с фильтрами."""
    nodes, _, _, _ = _data()
    df = nodes
    if role:
        df = df[df.role == role]
    if cluster_id is not None:
        df = df[df.cluster_id == cluster_id]
    df = df.nlargest(n, "priority_score")
    return {"nodes": df[["gid", "role", "priority_score", "cluster_id", "evidence"]].to_dict("records")}


def cluster_summary(cluster_id: int) -> dict:
    """Состав ролей, оборот, число seed и гипотеза по кластеру."""
    nodes, clusters, _, _ = _data()
    row = clusters[clusters.cluster_id == cluster_id]
    if row.empty:
        return {"error": f"кластер {cluster_id} не найден"}
    members = nodes[nodes.cluster_id == cluster_id]
    out = row.iloc[0].to_dict()
    out["role_counts"] = members.role.value_counts().to_dict()
    return out


def find_nodes(filters: dict, n: int = 10) -> dict:
    """Отбор по признакам, например {"taint_share": ">0.8", "out_deg": ">20"}."""
    nodes, _, _, _ = _data()
    df = nodes
    for col, cond in filters.items():
        if col not in df.columns:
            continue
        op, val = cond[:1], float(cond[1:]) if cond[1:2] != "=" else float(cond[2:])
        if cond.startswith(">="):
            df = df[df[col] >= val]
        elif cond.startswith("<="):
            df = df[df[col] <= val]
        elif op == ">":
            df = df[df[col] > val]
        elif op == "<":
            df = df[df[col] < val]
    df = df.nlargest(n, "priority_score")
    return {"matched": len(df), "nodes": df[["gid", "role", "priority_score", "evidence"]].to_dict("records")}


TOOL_REGISTRY = {
    "get_node": get_node,
    "who_collects_from": who_collects_from,
    "trace_path": trace_path,
    "top_nodes": top_nodes,
    "cluster_summary": cluster_summary,
    "find_nodes": find_nodes,
}
