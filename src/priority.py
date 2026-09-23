"""priority_score и топ-лист. Владелец: kuanyshs.

Композит из интерпретируемых слагаемых: топ-лист защищается по компонентам,
а не фразой «модель так решила».
"""

import numpy as np
import pandas as pd

WEIGHTS = {
    "flow": 0.30,       # сколько денег через узел прошло
    "external": 0.25,   # точка вливания средств извне выборки
    "ppr": 0.20,        # куда стекаются деньги фигурантов
    "transit": 0.15,    # чистота транзитного паттерна
    "seeds": 0.10,      # сколько фигурантов выше по течению
}


def _norm(s: pd.Series) -> pd.Series:
    """Ранговая нормировка: устойчива к выбросам, а их здесь много (max out_deg 116)."""
    return s.rank(pct=True).fillna(0.0)


def score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    flow = _norm(df.taint_share * df.out_kzt)
    external = _norm(df.external_funding_gap.clip(lower=0))
    ppr = _norm(df.ppr)
    fast = (df.median_delay_days >= 0) & (df.median_delay_days <= 2)
    transit = df.transit_ratio * np.where(fast, 1.0, 0.5)
    seeds = _norm(df.n_seeds_upstream)

    df["priority_score"] = (
        WEIGHTS["flow"] * flow
        + WEIGHTS["external"] * external
        + WEIGHTS["ppr"] * ppr
        + WEIGHTS["transit"] * transit
        + WEIGHTS["seeds"] * seeds
    ).clip(0, 1).round(4)
    return df


def top_nodes(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    top = df.nlargest(n, "priority_score").reset_index(drop=True)
    top["rank"] = top.index + 1
    top["why"] = [_why(r) for r in top.itertuples(index=False)]
    return top[["rank", "gid", "role", "priority_score", "why"]]


def _why(r) -> str:
    bits = [f"роль {r.role} (уверенность {r.role_score:.2f})"]
    if r.external_funding_gap > 0:
        bits.append(f"отдал на {r.external_funding_gap/1e6:.1f} млн больше, чем получил в графе — источник вне выборки")
    if r.out_deg >= 10:
        bits.append(f"рассылает на {r.out_deg} получателей")
    if r.in_deg >= 5:
        bits.append(f"получает от {r.in_deg} плательщиков")
    if r.transit_ratio >= 0.8:
        bits.append(f"пропускает насквозь (transit_ratio {r.transit_ratio:.2f})")
    if r.n_seeds_upstream >= 2:
        bits.append(f"деньги от {r.n_seeds_upstream} фигурантов")
    bits.append(f"taint {r.taint_share:.2f}")
    return "; ".join(bits)[:400]


def sanity(df: pd.DataFrame, k: int = 50) -> str:
    """Порог, меняющий решение: если топ забит seed и границей обхода,
    ранжирование поймало артефакт выгрузки, а не структуру сети."""
    top = df.nlargest(k, "priority_score")
    bad = ((top.is_seed) | (~top.was_expanded)).mean()
    verdict = "ОК" if bad <= 0.30 else "ВНИМАНИЕ: топ забит артефактами выгрузки"
    return f"доля seed/границы в топ-{k}: {bad*100:.0f}% — {verdict}"
