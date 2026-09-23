"""Кластеризация и гипотезы. Владелец: kuanyshs.

Две ступени: слабосвязные компоненты (честная, бесплатная структура)
и Louvain внутри крупнейшей. Louvain работает на НЕОРИЕНТИРОВАННОЙ проекции —
это оговаривается в README, стартовый код такое прямо разрешает.
"""

import networkx as nx
import pandas as pd

MIN_COMPONENT_TO_SPLIT = 500


def assign(df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """Проставляет cluster_id каждому узлу, включая изолированные."""
    cluster_of: dict[int, int] = {}
    cid = 0

    for comp in sorted(nx.weakly_connected_components(G), key=len, reverse=True):
        if len(comp) >= MIN_COMPONENT_TO_SPLIT:
            sub = G.subgraph(comp).to_undirected()
            for part in nx.community.louvain_communities(sub, weight="sum_kzt", seed=42):
                for n in part:
                    cluster_of[n] = cid
                cid += 1
        else:
            for n in comp:
                cluster_of[n] = cid
            cid += 1

    df = df.copy()
    df["cluster_id"] = df.gid.map(cluster_of)
    # Узлы без рёбер (19 seed-сирот) — каждый в свой кластер, чтобы не было пустого cluster_id
    orphans = df.cluster_id.isna()
    df.loc[orphans, "cluster_id"] = range(cid, cid + int(orphans.sum()))
    df["cluster_id"] = df.cluster_id.astype(int)
    return df


def summarize(df: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """clusters.csv: размер, seed, оборот и ГИПОТЕЗА по составу ролей."""
    cl = df.cluster_id
    internal = edges.merge(df[["gid", "cluster_id"]].rename(columns={"gid": "src", "cluster_id": "c_src"}), on="src")
    internal = internal.merge(df[["gid", "cluster_id"]].rename(columns={"gid": "dst", "cluster_id": "c_dst"}), on="dst")
    internal = internal[internal.c_src == internal.c_dst].groupby("c_src").sum_kzt.sum()

    rows = []
    for c, g in df.groupby(cl):
        rows.append({
            "cluster_id": int(c),
            "n_nodes": len(g),
            "n_seed": int(g.is_seed.sum()),
            "sum_kzt_internal": float(internal.get(c, 0.0)),
            "top_gids": ";".join(map(str, g.nlargest(3, "priority_score").gid)),
            "hypothesis": _hypothesis(g),
        })
    return pd.DataFrame(rows).sort_values("n_nodes", ascending=False).reset_index(drop=True)


def _hypothesis(g: pd.DataFrame) -> str:
    """Гипотеза выводится из состава ролей и потоков, а не из фантазии.

    Формулировки осторожные: ТЗ требует «признаки консолидации»,
    а не «преступная группа».
    """
    n = len(g)
    counts = g.role.value_counts()
    share = lambda r: counts.get(r, 0) / n

    if n <= 3:
        return "изолированный фрагмент, признаков структуры недостаточно"

    parts = []
    if share("consolidator") > 0.02 and share("peripheral") + share("terminal") > 0.5:
        parts.append("признаки воронки: много мелких плательщиков на несколько точек сбора")
    if share("distributor") > 0.05:
        parts.append("признаки веерного распределения средств")
    if share("transit") > 0.05:
        parts.append("признаки многошагового прогона через транзитные счета")
    if g.in_cycle_le6.any():
        parts.append(f"возвратные потоки: {int(g.in_cycle_le6.sum())} узлов в циклах")
    if g.external_share.mean() > 0.5:
        parts.append(f"{g.external_share.mean()*100:.0f}% оборота поступает извне выборки")

    if not parts:
        parts.append("выраженных структурных признаков не выявлено")
    return "; ".join(parts)[:400]
