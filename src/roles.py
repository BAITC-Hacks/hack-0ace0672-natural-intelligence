"""Правила ролей и role_score. Владелец: kuanyshs.

Пороги — перцентили НАШЕГО распределения, не константы из статей.
Каждый порог должен объясняться жюри за минуту.
"""

import networkx as nx
import numpy as np
import pandas as pd

# Пороги, которые калибруются. Держим в одном месте, чтобы README их процитировал.
P_HIGH_OUT_DEG = 0.90      # перцентиль out_deg для distributor
P_HIGH_IN_DEG = 0.90       # перцентиль in_deg для consolidator
TRANSIT_RATIO_MIN = 0.80   # вход и выход совпадают в пределах 20%
NET_FLOW_CONSOLIDATE = 0.30
NET_FLOW_DISTRIBUTE = -0.30
HHI_OUT_FAN = 0.30         # низкая концентрация получателей = настоящий веер
COORD_MIN_OUT_DEG = 5
P_COORD_TAINT = 0.75

ARTEFACT_PENALTY_BOUNDARY = 0.40  # depth=4: terminal может быть артефактом обхода
ARTEFACT_PENALTY_SEED = 0.30      # у seed входящие занижены по построению


def assign(df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """Возвращает df с колонками role, role_score, evidence."""
    q_out = df.loc[df.out_deg > 0, "out_deg"].quantile(P_HIGH_OUT_DEG)
    q_in = df.loc[df.in_deg > 0, "in_deg"].quantile(P_HIGH_IN_DEG)
    q_taint = df.taint_share.quantile(P_COORD_TAINT)

    has_edges = df.in_deg + df.out_deg > 0
    both = (df.in_deg > 0) & (df.out_deg > 0)

    # Правила проверяются сверху вниз, первое совпадение выигрывает.
    role = np.full(len(df), "peripheral", dtype=object)
    role[~has_edges.values] = "peripheral"

    is_transit = both & (df.transit_ratio >= TRANSIT_RATIO_MIN)
    is_distrib = (df.out_deg >= q_out) & (df.net_flow <= NET_FLOW_DISTRIBUTE) & (df.hhi_out < HHI_OUT_FAN)
    is_consol = (df.in_deg >= q_in) & (df.net_flow >= NET_FLOW_CONSOLIDATE)
    is_coord = (
        (df.n_seeds_upstream >= 2)
        & (df.out_deg >= COORD_MIN_OUT_DEG)
        & (df.taint_share >= q_taint)
        & (df.in_deg >= 2)
    )

    role[is_transit.values] = "transit"
    role[is_distrib.values] = "distributor"
    role[is_consol.values] = "consolidator"
    # coordinator назначается ПОСЛЕДНИМ среди активных ролей: узел, который
    # одновременно собирает и раздаёт деньги нескольких фигурантов, важнее
    # для аналитика, чем его частные признаки consolidator/distributor.
    role[is_coord.values] = "coordinator"
    # Тупики разбираем последними — они перекрывают всё остальное.
    role[(df.out_deg == 0).values & df.was_expanded.values] = "terminal"
    role[(df.out_deg == 0).values & (~df.was_expanded).values] = "terminal_unknown"
    role[~has_edges.values] = "peripheral"

    df = df.copy()
    df["role"] = role
    df["role_score"] = _score(df, q_out, q_in)
    df["evidence"] = _evidence(df, G)
    return df


def _score(df: pd.DataFrame, q_out: float, q_in: float) -> np.ndarray:
    """sigmoid(расстояние до порога) со штрафом за артефакт выгрузки.

    Даёт готовый ответ жюри на «почему уверенность 0.4, а не 0.9»
    и превращает скор из декорации в аргумент.
    """
    def sig(x, k=1.0):
        return 1.0 / (1.0 + np.exp(-k * x))

    base = np.full(len(df), 0.5)
    r = df.role.values

    base = np.where(r == "transit", sig((df.transit_ratio - TRANSIT_RATIO_MIN) * 10), base)
    base = np.where(r == "distributor", sig((df.out_deg / max(q_out, 1) - 1) * 3), base)
    base = np.where(r == "consolidator", sig((df.in_deg / max(q_in, 1) - 1) * 3), base)
    base = np.where(r == "coordinator", sig((df.n_seeds_upstream - 2) * 0.5 + df.taint_share), base)
    base = np.where(r == "terminal", sig((df.in_kzt > 0).astype(float) * 2), base)
    base = np.where(r == "terminal_unknown", 0.5, base)
    base = np.where(r == "peripheral", 1 - sig((df.in_deg + df.out_deg - 2) * 0.5), base)

    penalty = np.where(~df.was_expanded, ARTEFACT_PENALTY_BOUNDARY, 0.0)
    penalty = np.maximum(penalty, np.where(df.is_seed, ARTEFACT_PENALTY_SEED, 0.0))
    return np.clip(base * (1 - penalty), 0.0, 1.0).round(3)


def _evidence(df: pd.DataFrame, G: nx.DiGraph) -> list[str]:
    """<=200 символов, обязательно с числами. «Высокий скор» не принимается."""
    out = []
    for r in df.itertuples(index=False):
        m = f"{r.in_kzt/1e6:.1f}/{r.out_kzt/1e6:.1f} млн вх/исх"
        if r.role == "consolidator":
            s = f"получил {m} от {r.in_deg} плательщиков, отдал дальше {max(0,1-abs(r.net_flow))*100:.0f}%, HHI вх {r.hhi_in:.2f}"
        elif r.role == "distributor":
            s = f"разослал {r.out_kzt/1e6:.1f} млн на {r.out_deg} получателей, HHI исх {r.hhi_out:.2f}, {m}"
        elif r.role == "transit":
            d = "" if r.median_delay_days < 0 else f", задержка {r.median_delay_days:.0f} дн"
            s = f"вход~выход: {m}, transit_ratio {r.transit_ratio:.2f}{d}, taint {r.taint_share:.2f}"
        elif r.role == "coordinator":
            s = f"деньги от {r.n_seeds_upstream} фигурантов, {r.in_deg} плательщиков -> {r.out_deg} получателей, taint {r.taint_share:.2f}"
        elif r.role == "terminal":
            s = f"обход развернул узел, исходящих >=5000 KZT нет; получил {r.in_kzt/1e6:.1f} млн от {r.in_deg} плательщиков"
        elif r.role == "terminal_unknown":
            s = f"4-е колено, обход оборван: конечность НЕ подтверждена; получил {r.in_kzt/1e6:.1f} млн от {r.in_deg}"
        else:
            s = f"слабые связи: {r.in_deg} вх / {r.out_deg} исх контрагентов, {m}"
        out.append(s[:200])
    return out
