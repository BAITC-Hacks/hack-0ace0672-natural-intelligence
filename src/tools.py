"""Инструменты для ассистента: обычные функции на pandas. Владелец: narekdavtyan228-svg.

Они же используются шаблонным фолбэком, когда LLM недоступен.
Каждая функция возвращает dict, пригодный и для JSON, и для подстановки в шаблон.

Жёсткое правило трека: наружу отдаём только числа из выгрузок. Поэтому здесь нет
ни одной эвристики «на глаз» — всё, что возвращается, посчитано по out/*.csv и edges.
"""

import math
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

# Колонки, которые не несут смысла для аналитика и только удлиняют ответ модели.
_CARD_SKIP = {"was_expanded"}


@lru_cache(maxsize=1)
def _data():
    nodes = pd.read_csv(OUT / "nodes_roles.csv")
    clusters = pd.read_csv(OUT / "clusters.csv")
    top = pd.read_csv(OUT / "top_nodes.csv")
    edges = pd.read_parquet(ROOT / "data" / "edges.parquet")
    return nodes, clusters, top, edges


@lru_cache(maxsize=1)
def _graph():
    """Граф строится один раз на процесс: без кеша trace_path занимал 0.4 с на вызов."""
    import networkx as nx

    _, _, _, edges = _data()
    G = nx.DiGraph()
    for r in edges.itertuples(index=False):
        G.add_edge(int(r.src), int(r.dst), sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx))
    return G


def _clean(value):
    """NaN -> None, numpy -> python. Без этого json.dumps пишет литерал NaN,
    который не является валидным JSON и ломает разбор на стороне модели."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):  # numpy-скаляры
        return _clean(value.item())
    return value


def get_node(gid: int) -> dict:
    """Карточка узла: роль, уверенность, обоснование, ключевые числа, топ-контрагенты."""
    nodes, _, _, edges = _data()
    row = nodes[nodes.gid == gid]
    if row.empty:
        return {"error": f"gid {gid} не найден среди {len(nodes)} узлов выгрузки"}

    r = {k: v for k, v in row.iloc[0].to_dict().items() if k not in _CARD_SKIP}
    senders = edges[edges.dst == gid].nlargest(5, "sum_kzt")[["src", "sum_kzt", "n_tx"]]
    receivers = edges[edges.src == gid].nlargest(5, "sum_kzt")[["dst", "sum_kzt", "n_tx"]]
    r["top_senders"] = senders.to_dict("records")
    r["top_receivers"] = receivers.to_dict("records")
    return _clean(r)


def who_collects_from(gids: list[int], max_depth: int = 3, n: int = 10) -> dict:
    """Кто собирает деньги с указанных узлов — получатели ниже по течению.

    Прямой ответ на сценарий ТЗ: «кто собирает деньги с этих пятерых?».

    Помимо достижимости считаем СУММЫ: вопрос задан про деньги, а не про связность.
    `received_kzt` — сколько пришло на узел по рёбрам, исходящим из прослеженной
    области (сами указанные узлы и всё, что достижимо от них за max_depth шагов).
    """
    nodes, _, _, edges = _data()
    if not gids:
        return {"error": "не указан ни один gid"}

    known = set(nodes.gid)
    unknown = [g for g in gids if g not in known]
    gids = [g for g in gids if g in known]
    if not gids:
        return {"error": f"ни один из gid не найден в выгрузке: {unknown}"}

    reach: dict[int, set[int]] = {}
    for g in gids:
        seen: set[int] = set()
        frontier = {g}
        for _ in range(max_depth):
            frontier = set(edges[edges.src.isin(frontier)].dst) - seen
            if not frontier:
                break
            seen |= frontier
        reach[g] = seen

    common: dict[int, int] = {}
    for s in reach.values():
        for node in s:
            common[node] = common.get(node, 0) + 1
    if not common:
        return {"asked_gids": gids, "collectors": [],
                "note": "от указанных узлов не исходит ни одного перевода в выгрузке"}

    # Деньги, пришедшие на получателя именно из прослеженной области.
    traced = set(gids) | set(common)
    inflow = (
        edges[edges.dst.isin(common) & edges.src.isin(traced)]
        .groupby("dst")
        .agg(received_kzt=("sum_kzt", "sum"), n_tx=("n_tx", "sum"), from_traced=("src", "nunique"))
    )

    res = (
        nodes[nodes.gid.isin(common)]
        .assign(
            reached_from=lambda d: d.gid.map(common),
            received_kzt=lambda d: d.gid.map(inflow.received_kzt).fillna(0.0),
            n_tx=lambda d: d.gid.map(inflow.n_tx).fillna(0).astype(int),
            from_traced=lambda d: d.gid.map(inflow.from_traced).fillna(0).astype(int),
        )
        .sort_values(["reached_from", "received_kzt"], ascending=False)
        .head(n)
    )

    out = {
        "asked_gids": gids,
        "max_depth": max_depth,
        "reachable_total": len(common),
        "collectors": res[["gid", "role", "priority_score", "reached_from",
                           "received_kzt", "n_tx", "from_traced", "evidence"]].to_dict("records"),
    }
    if unknown:
        out["not_found"] = unknown
    return _clean(out)


def trace_path(src: int, dst: int, max_len: int = 6) -> dict:
    """Путь движения денег между двумя узлами: до пяти простых путей с суммами."""
    import networkx as nx

    G = _graph()
    if src not in G:
        return {"error": f"gid {src} не участвует ни в одном переводе выгрузки"}
    if dst not in G:
        return {"error": f"gid {dst} не участвует ни в одном переводе выгрузки"}

    found = []
    for path in nx.all_simple_paths(G, src, dst, cutoff=max_len):
        hops = [
            {"src": u, "dst": v, "sum_kzt": G[u][v]["sum_kzt"], "n_tx": G[u][v]["n_tx"]}
            for u, v in zip(path, path[1:])
        ]
        found.append({
            "path": path,
            "length": len(path) - 1,
            # Узкое место: больше этой суммы по пути пройти не могло.
            "min_edge_kzt": min(h["sum_kzt"] for h in hops),
            "hops": hops,
        })
        if len(found) == 5:
            break

    return _clean({"src": src, "dst": dst, "max_len": max_len,
                   "found": len(found), "paths": found})


def top_nodes(role: str | None = None, cluster_id: int | None = None, n: int = 10) -> dict:
    """Топ по приоритету проверки с фильтрами по роли и кластеру."""
    nodes, _, _, _ = _data()
    df = nodes
    if role:
        known_roles = sorted(nodes.role.unique())
        if role not in known_roles:
            return {"error": f"роль '{role}' не из словаря", "known_roles": known_roles}
        df = df[df.role == role]
    if cluster_id is not None:
        df = df[df.cluster_id == cluster_id]
    matched = len(df)
    df = df.nlargest(n, "priority_score")
    return _clean({
        "matched": matched,
        "returned": len(df),
        "nodes": df[["gid", "role", "priority_score", "cluster_id", "evidence"]].to_dict("records"),
    })


def cluster_summary(cluster_id: int) -> dict:
    """Состав ролей, оборот, число seed и гипотеза по кластеру."""
    nodes, clusters, _, _ = _data()
    row = clusters[clusters.cluster_id == cluster_id]
    if row.empty:
        return {"error": f"кластер {cluster_id} не найден",
                "known_clusters": f"{clusters.cluster_id.min()}..{clusters.cluster_id.max()}"}

    members = nodes[nodes.cluster_id == cluster_id]
    out = row.iloc[0].to_dict()
    out["role_counts"] = members.role.value_counts().to_dict()
    out["top_by_priority"] = (
        members.nlargest(5, "priority_score")[["gid", "role", "priority_score"]].to_dict("records")
    )
    return _clean(out)


# Разбор условий вида ">0.8", ">=20", "<1e6", "=transit", "!=terminal".
_COND = re.compile(r"^\s*(>=|<=|!=|==|=|>|<)?\s*(.+?)\s*$")


def find_nodes(filters: dict, n: int = 10) -> dict:
    """Отбор по признакам, например {"taint_share": ">0.8", "out_deg": ">20"}.

    Условие без оператора считается равенством. Нечисловые значения сравниваются
    как строки — это позволяет фильтровать и по role.
    """
    nodes, _, _, _ = _data()
    if not isinstance(filters, dict) or not filters:
        return {"error": "filters пуст", "known_columns": sorted(nodes.columns)}

    df = nodes
    applied, ignored = {}, {}

    for col, cond in filters.items():
        if col not in df.columns:
            ignored[col] = "нет такой колонки"
            continue

        m = _COND.match(str(cond))
        if not m:
            ignored[col] = f"не разобрано условие {cond!r}"
            continue
        op, raw = m.group(1) or "==", m.group(2)

        try:
            val = float(raw)
        except ValueError:
            val = raw.strip("'\"")
            if op not in ("==", "=", "!="):
                ignored[col] = f"оператор {op} неприменим к строке {val!r}"
                continue

        series = df[col]
        if isinstance(val, float) and not pd.api.types.is_numeric_dtype(series):
            ignored[col] = "колонка нечисловая"
            continue

        if op == ">=":
            df = df[series >= val]
        elif op == "<=":
            df = df[series <= val]
        elif op == ">":
            df = df[series > val]
        elif op == "<":
            df = df[series < val]
        elif op == "!=":
            df = df[series.astype(str) != str(val)] if isinstance(val, str) else df[series != val]
        else:
            df = df[series.astype(str) == str(val)] if isinstance(val, str) else df[series == val]
        applied[col] = f"{op}{val}"

    if not applied:
        return {"error": "ни одно условие не применено", "ignored": ignored,
                "known_columns": sorted(nodes.columns)}

    # matched считается ДО обрезки: иначе ассистент назовёт жюри размер выборки вместо
    # числа совпадений и соврёт на ровном месте.
    matched = len(df)
    df = df.nlargest(n, "priority_score")

    out = {
        "applied": applied,
        "matched": matched,
        "returned": len(df),
        "nodes": df[["gid", "role", "priority_score", "evidence"]].to_dict("records"),
    }
    if ignored:
        out["ignored"] = ignored
    return _clean(out)


def network_resilience(removed: int | None = None) -> dict:
    """Что будет с сетью, если изъять топ-N узлов по приоритету.

    Опциональный пункт ТЗ. Рядом всегда идёт случайное изъятие такого же числа узлов:
    без базиса цифра «171 компонента» ни о чём не говорит.
    """
    path = OUT / "resilience.csv"
    if not path.exists():
        return {"error": "out/resilience.csv нет — запустите python run.py"}

    df = pd.read_csv(path)
    if removed is not None:
        df = df[df.removed == removed]
        if df.empty:
            return {"error": f"шага removed={removed} нет",
                    "known_steps": pd.read_csv(path).removed.tolist()}

    rows = []
    for r in df.itertuples(index=False):
        rows.append({
            "removed": int(r.removed),
            "components_targeted": int(r.components_targeted),
            "components_random": float(r.components_random),
            "largest_targeted": int(r.largest_targeted),
            "largest_random": float(r.largest_random),
            "isolated_targeted": int(r.isolated_targeted),
            "ratio_vs_random": round(r.components_targeted / r.components_random, 1),
        })
    return _clean({
        "steps": rows,
        "note": "components_random — среднее по 5 прогонам случайного изъятия того же числа "
                "узлов. Это проверка осмысленности топ-листа, а НЕ рекомендация блокировать счета.",
    })


TOOL_REGISTRY = {
    "get_node": get_node,
    "who_collects_from": who_collects_from,
    "trace_path": trace_path,
    "top_nodes": top_nodes,
    "cluster_summary": cluster_summary,
    "find_nodes": find_nodes,
    "network_resilience": network_resilience,
}
