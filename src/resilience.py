"""Устойчивость сети: что будет, если изъять топ-N узлов. Владелец: kuanyshs.

Опциональный пункт ТЗ: «что произойдёт с сетью при изъятии топ-N узлов;
распадается ли она на изолированные фрагменты».

Операционный смысл прямой: если блокировка 10 счетов рвёт сеть на куски,
приоритизация нашла настоящие узлы связности, а не просто крупные обороты.
Для сравнения рядом считаем изъятие случайных узлов — без него цифра
ни о чём не говорит.
"""

import networkx as nx
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def report(df: pd.DataFrame, G: nx.DiGraph, steps=(5, 10, 20, 50)) -> pd.DataFrame:
    """Сравнивает изъятие топ-N по приоритету с изъятием случайных N узлов."""
    base_comp = nx.number_weakly_connected_components(G)
    base_big = len(max(nx.weakly_connected_components(G), key=len))
    order = df.sort_values("priority_score", ascending=False).gid.tolist()
    order = [g for g in order if g in G]
    pool = [n for n in G.nodes]

    rows = []
    for k in steps:
        targeted = _cut(G, order[:k])
        random_runs = [_cut(G, RNG.choice(pool, size=k, replace=False).tolist()) for _ in range(5)]
        rnd_comp = float(np.mean([r[0] for r in random_runs]))
        rnd_big = float(np.mean([r[1] for r in random_runs]))
        rows.append({
            "removed": k,
            "components_targeted": targeted[0],
            "components_random": round(rnd_comp, 1),
            "largest_targeted": targeted[1],
            "largest_random": round(rnd_big, 1),
            "isolated_targeted": targeted[2],
        })

    out = pd.DataFrame(rows)
    out.attrs["base_components"] = base_comp
    out.attrs["base_largest"] = base_big
    return out


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
        f"{'изъято':>7} {'компонент':>11} {'(случайно)':>11} "
        f"{'крупнейшая':>11} {'(случайно)':>11} {'одиночек':>9}",
    ]
    for r in out.itertuples(index=False):
        lines.append(
            f"{r.removed:>7} {r.components_targeted:>11} {r.components_random:>11} "
            f"{r.largest_targeted:>11} {r.largest_random:>11} {r.isolated_targeted:>9}"
        )
    return "\n".join(lines)
