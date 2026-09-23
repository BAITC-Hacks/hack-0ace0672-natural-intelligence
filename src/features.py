"""Признаки узла. Владелец: kuanyshs.

Базовый набор реализован, пороги и веса калибруются в roles.py / priority.py.
Всё считается за ~1.5 с на полном графе — лимит ТЗ в 5 минут не связывает.
"""

import networkx as nx
import numpy as np
import pandas as pd


def compute(G: nx.DiGraph, nodes: pd.DataFrame, tx: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    """Единая точка входа: возвращает DataFrame на все 2248 узлов."""
    df = nodes[["gid", "depth", "is_seed"]].copy()
    df = _degrees(df, G)
    df = _ratios(df)
    df = _concentration(df, G)
    df["taint_share"] = df.gid.map(taint_haircut(G, seeds)).fillna(0.0)
    df.loc[df.is_seed, "taint_share"] = 1.0  # в т.ч. 19 seed-сирот, которых нет в G
    # Доля оборота, пришедшая ИЗВНЕ выборки. Именно она различает узлы:
    # taint высок почти у всех, потому что граф построен обходом от seed.
    df["external_share"] = 1.0 - df["taint_share"]
    df["ppr"] = df.gid.map(personalized_pagerank(G, seeds)).fillna(0.0)
    df["n_seeds_upstream"] = df.gid.map(n_seeds_upstream(G, seeds)).fillna(0).astype(int)
    df = _temporal(df, tx)
    df = _structural(df, G)
    return df


# ------------------------------------------------------------------ базовые

def _degrees(df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    maps = {
        "in_deg": dict(G.in_degree()),
        "out_deg": dict(G.out_degree()),
        "in_kzt": dict(G.in_degree(weight="sum_kzt")),
        "out_kzt": dict(G.out_degree(weight="sum_kzt")),
        "in_tx": dict(G.in_degree(weight="n_tx")),
        "out_tx": dict(G.out_degree(weight="n_tx")),
    }
    for col, m in maps.items():
        df[col] = df.gid.map(m).fillna(0)
    for col in ("in_deg", "out_deg", "in_tx", "out_tx"):
        df[col] = df[col].astype(int)
    return df


def _ratios(df: pd.DataFrame) -> pd.DataFrame:
    lo = np.minimum(df.in_kzt, df.out_kzt)
    hi = np.maximum(df.in_kzt, df.out_kzt)
    # Ограничен [0,1] и не взрывается на seed, в отличие от out/in из стартового кода.
    df["transit_ratio"] = np.where(hi > 0, lo / hi, 0.0)
    total = df.in_kzt + df.out_kzt
    # Знак разделяет: >0 копит (consolidator), <0 раздаёт (distributor).
    df["net_flow"] = np.where(total > 0, (df.in_kzt - df.out_kzt) / total, 0.0)
    # 335 не-seed узлов отдают больше, чем получили: 229.9 млн = 63% оборота извне выборки.
    df["external_funding_gap"] = df.out_kzt - df.in_kzt
    # Обход разворачивал узел и исходящих >=5000 KZT не нашёл — значит сток настоящий.
    df["was_expanded"] = df.depth <= 3
    return df


def _concentration(df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """Херфиндаль по долям контрагентов: высокий = воронка, низкий = веер."""

    def hhi(gid, direction):
        edges = G.in_edges(gid, data=True) if direction == "in" else G.out_edges(gid, data=True)
        w = np.array([d["sum_kzt"] for *_, d in edges], dtype=float)
        if w.sum() <= 0:
            return 0.0
        return float(((w / w.sum()) ** 2).sum())

    present = set(G.nodes)
    df["hhi_in"] = [hhi(g, "in") if g in present else 0.0 for g in df.gid]
    df["hhi_out"] = [hhi(g, "out") if g in present else 0.0 for g in df.gid]
    return df


# ------------------------------------------------------- происхождение денег

def taint_haircut(G: nx.DiGraph, seeds: list[int], iters: int = 12) -> dict[int, float]:
    """Haircut-трассировка: какая доля входящих денег узла пришла от seed.

    Seed = 1.0 по определению. Остальные получают средневзвешенную по суммам
    долю taint своих плательщиков. Граф не DAG (177 взаимных пар, тысячи циклов),
    поэтому считаем итеративно до сходимости, а не топологическим порядком.

    Это прямой ответ на вопрос кейса «чьи это деньги»: узел с 99 получателями
    и taint 0.09 — внешний бизнес, узел с 61 получателем и taint 0.53 —
    распределитель денег группы. Степени этого не различают.
    """
    seed_set = set(seeds)
    taint = {n: (1.0 if n in seed_set else 0.0) for n in G.nodes}

    for _ in range(iters):
        nxt = {}
        for n in G.nodes:
            if n in seed_set:
                nxt[n] = 1.0
                continue
            num = 0.0
            den = 0.0
            for u, _, d in G.in_edges(n, data=True):
                w = d["sum_kzt"]
                num += taint[u] * w
                den += w
            nxt[n] = num / den if den > 0 else 0.0
        if max(abs(nxt[n] - taint[n]) for n in G.nodes) < 1e-6:
            taint = nxt
            break
        taint = nxt
    return taint


def personalized_pagerank(G: nx.DiGraph, seeds: list[int]) -> dict[int, float]:
    """PageRank с рестартом на seed.

    Обычный PageRank на этом графе полуосмыслен: 1554 узла без исходящих,
    телепорт размазывает вес по всей выборке. Рестарт на 81 seed отвечает
    на вопрос кейса буквально — куда стекаются деньги фигурантов.
    """
    p = {s: 1.0 for s in seeds if s in G}
    if not p:
        return {}
    return nx.pagerank(G, personalization=p, weight="sum_kzt")


def n_seeds_upstream(G: nx.DiGraph, seeds: list[int]) -> dict[int, int]:
    """От скольких РАЗНЫХ фигурантов деньги доходят до узла.

    ВНИМАНИЕ: признак плохо ранжирует — 1134 узла имеют ровно 7, 563 узла ровно 1.
    Использовать как фильтр (>=2), а не как основу приоритета.
    """
    counts: dict[int, int] = {n: 0 for n in G.nodes}
    for s in seeds:
        if s not in G:
            continue
        for n in nx.descendants(G, s) | {s}:
            counts[n] += 1
    return counts


# ------------------------------------------------------------------ время

def _temporal(df: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Транзит, подтверждённый по времени, а не только по совпадению сумм."""
    first_in = tx.groupby("dst").date.min()
    first_out = tx.groupby("src").date.min()

    df["sent_before_received"] = df.gid.map(
        lambda g: bool(g in first_out.index and (g not in first_in.index or first_out[g] < first_in[g]))
    )

    # медиана задержки между первым входящим и последующими исходящими
    delay = {}
    out_by_src = {k: v.sort_values() for k, v in tx.groupby("src").date}
    for g, fin in first_in.items():
        outs = out_by_src.get(g)
        if outs is None:
            continue
        after = outs[outs >= fin]
        if len(after):
            delay[g] = float((after - fin).dt.days.median())
    df["median_delay_days"] = df.gid.map(delay).fillna(-1.0)

    # синхронные переводы: сколько раз узел получал от >=3 разных отправителей за один день
    ev = tx.groupby(["dst", "date"]).src.nunique()
    sync = ev[ev >= 3].groupby(level=0).size()
    df["sync_in_events"] = df.gid.map(sync).fillna(0).astype(int)
    return df


# -------------------------------------------------------------- структура

def _structural(df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """Дробление и возвратные потоки."""
    max_tx = {}
    for u, v, d in G.edges(data=True):
        max_tx[u] = max(max_tx.get(u, 0), d["n_tx"])
        max_tx[v] = max(max_tx.get(v, 0), d["n_tx"])
    df["max_edge_n_tx"] = df.gid.map(max_tx).fillna(0).astype(int)

    # length_bound обязателен: граф не DAG, без него уйдём за лимит времени
    in_cycle: set[int] = set()
    for cyc in nx.simple_cycles(G, length_bound=6):
        in_cycle.update(cyc)
    df["in_cycle_le6"] = df.gid.isin(in_cycle)
    return df
