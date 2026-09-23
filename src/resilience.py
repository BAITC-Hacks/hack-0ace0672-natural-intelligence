"""Устойчивость сети: что будет, если изъять топ-N узлов. Владелец: kuanyshs.

Опциональный пункт ТЗ: «что произойдёт с сетью при изъятии топ-N узлов;
распадается ли она на изолированные фрагменты».

ДВЕ базы сравнения, и вторая обязательна.

Первая версия сравнивала только со случайными узлами и показывала разрыв в
десять раз: 171 компонента против 16.6. Это ничего не доказывало. На графе,
где медиана степени равна 2, а максимум 116, изъятие ЛЮБОГО хаба разносит сеть,
и сравнение с равномерно случайным узлом отвечает на вопрос «важна ли степень»,
а не на вопрос «нашли ли мы что-то сверх степени».

Правильная база — узлы ТОЙ ЖЕ СТЕПЕНИ. Она и показала, что отдельного эффекта
у нашей приоритизации нет: случайные узлы сопоставимой степени рвут сеть не хуже.
Это честный отрицательный результат, и он остаётся в выводе.
"""

import networkx as nx
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
DEGREE_TOLERANCE = 0.2   # «та же степень» = в пределах ±20%
REPEATS = 10             # прогонов на каждую случайную базу


def report(df: pd.DataFrame, G: nx.DiGraph, steps=(5, 10, 20, 50)) -> pd.DataFrame:
    base_comp = nx.number_weakly_connected_components(G)
    base_big = len(max(nx.weakly_connected_components(G), key=len))

    order = [g for g in df.sort_values("priority_score", ascending=False).gid if g in G]
    pool = list(G.nodes)
    degree = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes}

    rows = []
    for k in steps:
        victims = order[:k]
        tgt = _cut(G, victims)
        unif = _mean_runs(G, lambda: RNG.choice(pool, size=k, replace=False).tolist())
        matched = _mean_runs(G, lambda: _degree_matched(victims, degree))
        rows.append({
            "removed": k,
            "components_targeted": tgt[0],
            "components_random": round(unif[0], 1),
            "components_same_degree": round(matched[0], 1),
            "largest_targeted": tgt[1],
            "largest_random": round(unif[1], 1),
            "largest_same_degree": round(matched[1], 1),
            "isolated_targeted": tgt[2],
        })

    out = pd.DataFrame(rows)
    out.attrs["base_components"] = base_comp
    out.attrs["base_largest"] = base_big
    return out


def _degree_matched(victims: list[int], degree: dict[int, int]) -> list[int]:
    """Для каждого изымаемого узла — случайный узел сопоставимой степени."""
    chosen: list[int] = []
    for v in victims:
        d = degree[v]
        lo, hi = d * (1 - DEGREE_TOLERANCE), d * (1 + DEGREE_TOLERANCE)
        pool = [n for n, dn in degree.items() if lo <= dn <= hi and n not in chosen]
        chosen.append(RNG.choice(pool) if pool else v)
    return chosen


def _mean_runs(G: nx.DiGraph, sampler) -> tuple[float, float]:
    comps, bigs = [], []
    for _ in range(REPEATS):
        c, b, _ = _cut(G, sampler())
        comps.append(c)
        bigs.append(b)
    return float(np.mean(comps)), float(np.mean(bigs))


def _cut(G: nx.DiGraph, victims: list[int]) -> tuple[int, int, int]:
    H = G.copy()
    H.remove_nodes_from(victims)
    if H.number_of_nodes() == 0:
        return 0, 0, 0
    comps = list(nx.weakly_connected_components(H))
    return len(comps), len(max(comps, key=len)), sum(1 for c in comps if len(c) == 1)


def render(out: pd.DataFrame) -> str:
    lines = [
        f"исходно: {out.attrs['base_components']} компонент, "
        f"крупнейшая {out.attrs['base_largest']} узлов",
        f"{'изъято':>7} {'наш топ':>9} {'случайные':>11} {'та же степень':>15}"
        f" | {'крупнейшая':>11} {'та же степень':>15}",
    ]
    for r in out.itertuples(index=False):
        lines.append(
            f"{r.removed:>7} {r.components_targeted:>9} {r.components_random:>11} "
            f"{r.components_same_degree:>15} | {r.largest_targeted:>11} "
            f"{r.largest_same_degree:>15}"
        )
    lines.append("\n" + verdict(out))
    return "\n".join(lines)


def verdict(out: pd.DataFrame) -> str:
    """Честный вывод, а не тот, который хотелось бы получить.

    Считаем по ВСЕМ шагам: на одном значении N перевес легко получить случайно.
    """
    ratio = (out.components_targeted / out.components_same_degree).mean()
    vs_random = (out.components_targeted / out.components_random).mean()

    if ratio >= 1.2:
        return ("  ВЫВОД: наш топ разрушает сеть в "
                f"{ratio:.2f} раза сильнее узлов той же степени — "
                "приоритизация нашла точки связности сверх степени")
    return (
        f"  ВЫВОД: в среднем по шагам наш топ даёт в {ratio:.2f} раза "
        "компонент относительно случайных узлов ТОЙ ЖЕ СТЕПЕНИ, то есть\n"
        "  разрушительность объясняется степенью узла; отдельного эффекта\n"
        "  приоритизации не обнаружено. Против равномерно случайных узлов перевес\n"
        f"  выглядит внушительно ({vs_random:.1f}x), но эта база отвечает на вопрос\n"
        "  «важна ли степень», а не «нашли ли мы что-то сверх неё», и приводить её\n"
        "  как доказательство было бы подменой."
    )
