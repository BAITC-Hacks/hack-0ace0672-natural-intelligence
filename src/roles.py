"""Правила ролей и role_score. Владелец: kuanyshs.

Пороги — перцентили НАШЕГО распределения, не константы из статей.
Каждый порог должен объясняться жюри за минуту.
"""

import networkx as nx
import numpy as np
import pandas as pd

# Пороги. Держим в одном месте, чтобы README их процитировал без раскопок по коду.
#
# Пороги заданы числами, а не перцентилями, потому что на этом графе перцентили
# вырождаются: медиана in_deg равна 1, поэтому P90 = 2 и правило пропускает почти всех.
# Рядом с каждым числом указано, какой доле распределения оно соответствует —
# именно это и объясняется жюри.

MIN_IN_DEG_CONSOLIDATOR = 4    # >=4 плательщиков: верхние ~3% узлов с входящими
MIN_OUT_DEG_DISTRIBUTOR = 10   # >=10 получателей: верхние ~7% узлов с исходящими
TRANSIT_RATIO_MIN = 0.80       # вход и выход совпадают в пределах 20%
NET_FLOW_CONSOLIDATE = 0.30    # удержал >=65% того, что получил
NET_FLOW_DISTRIBUTE = -0.30    # отдал заметно больше, чем получил внутри графа
HHI_OUT_FAN = 0.30             # низкая концентрация получателей = настоящий веер
COORD_MIN_IN_DEG = 4           # координатор одновременно собирает...
COORD_MIN_OUT_DEG = 5          # ...и раздаёт
COORD_MIN_SEEDS = 2            # деньги как минимум двух разных фигурантов
TURNOVER_PERCENTILE = 0.90     # верхние 10% по обороту среди узлов с входом и выходом
GAP_PERCENTILE = 0.50          # медиана разрыва «отдал минус получил» среди тех, у кого он есть

ARTEFACT_PENALTY_BOUNDARY = 0.40  # depth=4: terminal может быть артефактом обхода
ARTEFACT_PENALTY_SEED = 0.30      # у seed входящие занижены по построению


def assign(df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """Возвращает df с колонками role, role_score, evidence."""
    has_edges = df.in_deg + df.out_deg > 0
    both = (df.in_deg > 0) & (df.out_deg > 0)

    # Узел попадает в роль либо по ЧИСЛУ контрагентов, либо по ОБЪЁМУ.
    # Без второго пути узлы с двумя контрагентами, но миллионным потоком,
    # сваливались в peripheral («признаков роли не выявлено») и при этом
    # занимали верх топа приоритета — прямое противоречие, которое заметит жюри.
    turnover = df.in_kzt + df.out_kzt
    big_money = turnover >= turnover[both].quantile(TURNOVER_PERCENTILE)

    # Заметный разрыв «отдал минус получил» — это точка вливания средств извне выборки.
    # Такой узел не может считаться периферией: у него ЕСТЬ структурный признак,
    # просто он виден не по числу контрагентов, а по происхождению денег.
    gaps = df.loc[both & (df.external_funding_gap > 0), "external_funding_gap"]
    gap_thr = gaps.quantile(GAP_PERCENTILE) if len(gaps) else float("inf")
    big_gap = both & (df.external_funding_gap >= gap_thr)

    # Правила проверяются сверху вниз, первое совпадение выигрывает.
    role = np.full(len(df), "peripheral", dtype=object)
    role[~has_edges.values] = "peripheral"

    is_transit = both & (df.transit_ratio >= TRANSIT_RATIO_MIN)
    # Веер определяется числом получателей и низкой концентрацией, а НЕ тем,
    # удержал ли узел часть денег. Проверка planted-injection показала: узел,
    # принявший 4 млн и разославший 4.5 млн на 20 получателей, имеет net_flow
    # около нуля и при старом условии (net_flow <= -0.3) уходил в transit —
    # ни один внедрённый веер не опознавался. Требуем лишь, чтобы узел
    # не был накопителем.
    is_distrib = (
        (df.out_deg >= MIN_OUT_DEG_DISTRIBUTOR)
        & (df.net_flow < NET_FLOW_CONSOLIDATE)
        & (df.hhi_out < HHI_OUT_FAN)
    )
    is_consol = (df.in_deg >= MIN_IN_DEG_CONSOLIDATOR) & (df.net_flow >= NET_FLOW_CONSOLIDATE)

    # Слабое назначение по направлению потока для узлов из верхних 10% по обороту.
    # Крупный поток — уже признак, даже если контрагент всего один. Такие узлы
    # получают роль с НИЗКИМ role_score: признак есть, но слабый.
    notable = (big_money | big_gap) & both
    weak_consol = notable & (df.net_flow >= 0)
    weak_distrib = notable & (df.net_flow < 0)
    # Условие по taint_share намеренно НЕ используется: у 75% узлов он равен ровно 1.0,
    # потому что граф построен обходом от seed. Порог по нему вырождается в "taint == 1".
    is_coord = (
        both
        & (df.n_seeds_upstream >= COORD_MIN_SEEDS)
        & (
            ((df.in_deg >= COORD_MIN_IN_DEG) & (df.out_deg >= COORD_MIN_OUT_DEG))
            | (big_money & (df.in_deg >= 3) & (df.out_deg >= 3))
        )
    )

    # Сначала слабые назначения по обороту, поверх них — специфические паттерны.
    role[weak_distrib.values] = "distributor"
    role[weak_consol.values] = "consolidator"
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
    df["role_score"] = _score(df)
    df["evidence"] = _evidence(df, G)
    return df


def _score(df: pd.DataFrame) -> np.ndarray:
    """sigmoid(расстояние до порога) со штрафом за артефакт выгрузки.

    Даёт готовый ответ жюри на «почему уверенность 0.4, а не 0.9»
    и превращает скор из декорации в аргумент.
    """
    def sig(x, k=1.0):
        return 1.0 / (1.0 + np.exp(-k * x))

    base = np.full(len(df), 0.5)
    r = df.role.values

    base = np.where(r == "transit", sig((df.transit_ratio - TRANSIT_RATIO_MIN) * 10), base)
    base = np.where(r == "distributor", sig((df.out_deg / MIN_OUT_DEG_DISTRIBUTOR - 1) * 3), base)
    base = np.where(r == "consolidator", sig((df.in_deg / MIN_IN_DEG_CONSOLIDATOR - 1) * 4), base)
    base = np.where(
        r == "coordinator",
        sig((df.in_deg / COORD_MIN_IN_DEG - 1) * 2 + (df.out_deg / COORD_MIN_OUT_DEG - 1) * 2),
        base,
    )
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
        m = f"{_kzt(r.in_kzt)} вх / {_kzt(r.out_kzt)} исх"
        if r.role == "consolidator":
            kept = 100 * (1 - r.out_kzt / r.in_kzt) if r.in_kzt > 0 else 100
            s = (f"получил {_kzt(r.in_kzt)} от {r.in_deg} плательщиков, "
                 f"удержал {kept:.0f}%, HHI вх {r.hhi_in:.2f}")
        elif r.role == "distributor":
            ext = (f", из них {_kzt(r.external_funding_gap)} пришло извне выборки"
                   if r.external_funding_gap > 0 else "")
            s = f"разослал {_kzt(r.out_kzt)} на {r.out_deg} получателей{ext}, HHI исх {r.hhi_out:.2f}"
        elif r.role == "transit":
            d = "" if r.median_delay_days < 0 else f", задержка {r.median_delay_days:.0f} дн"
            s = f"прошло насквозь: {m}, transit_ratio {r.transit_ratio:.2f}{d}"
        elif r.role == "coordinator":
            s = (f"деньги от {r.n_seeds_upstream} фигурантов; {r.in_deg} плательщиков -> "
                 f"{r.out_deg} получателей, {m}")
        elif r.role == "terminal":
            s = (f"обход развернул узел, исходящих >=5000 KZT нет; "
                 f"получил {_kzt(r.in_kzt)} от {r.in_deg} плательщиков")
        elif r.role == "terminal_unknown":
            s = (f"4-е колено, обход оборван: конечность НЕ подтверждена; "
                 f"получил {_kzt(r.in_kzt)} от {r.in_deg} плательщиков")
        elif r.in_deg == 0 and r.out_deg == 0:
            s = "нет ни одного перевода >=5000 KZT в выгрузке — узел изолирован"
        else:
            s = f"слабые связи: {r.in_deg} вх / {r.out_deg} исх контрагентов, {m}"
        out.append(s[:200])
    return out


def _kzt(v: float) -> str:
    """Суммы читаемо: миллионы для крупных, тысячи для мелких.

    Единый формат «0.0 млн» превращал мелкие суммы в ноль и делал evidence бесполезным.
    """
    if v >= 1e6:
        return f"{v/1e6:.1f} млн"
    if v >= 1e3:
        return f"{v/1e3:.0f} тыс"
    return f"{v:.0f}"
