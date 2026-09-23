"""Структурные мотивы: улика вместо метрики. Владелец: kuanyshs.

Степени и PageRank дают ЧИСЛО. Аналитику нужен проверяемый ФАКТ — конкретная
пара платежей, конкретный цикл с датами, конкретное пересечение получателей.
Это прямо работает на must-have №3: жюри называет gid и просит объяснить роль.

Модуль АДДИТИВНЫЙ: роли, пороги, priority_score и кластеры он не трогает.
Добавляет четыре колонки в nodes_roles.csv и пишет out/motifs.csv.

У каждого детектора есть контрольное условие, без которого совпадение
было бы артефактом размера узла:
  циклы        -> хронология переводов
  pass-through -> допуск по сумме и времени
  пересечения  -> конфигурационная модель
"""

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

CYCLE_MAX_LEN = 4          # длиннее не берём: рост комбинаторный, отдача падает
AMOUNT_TOLERANCE = 0.03    # 3% — банковская комиссия и округление укладываются
DELAY_DAYS = 2             # окно «пришло и почти сразу ушло»
MIN_OUT_DEG_PAIR = 5       # ниже этой степени пересечение получателей неинформативно
MIN_OVERLAP = 5            # ниже пяти общих получателей z вырождается: при ожидании
                           # 0.03 любое совпадение даёт z=17, что ничего не значит
MIN_EXPECTED = 0.5         # z считаем только там, где случайное ожидание осмысленно
SIMULATIONS = 2000         # прогонов конфигурационной модели

RNG = np.random.default_rng(42)


def compute(G: nx.DiGraph, tx: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (df с четырьмя новыми колонками, таблица пар для out/motifs.csv)."""
    df = df.copy()

    in_cycle = time_respecting_cycles(G, tx)
    df["cycle_time_ok"] = df.gid.isin(in_cycle)

    matches = passthrough_matches(tx)
    df["passthrough_matches"] = df.gid.map(matches).fillna(0).astype(int)

    pairs = shared_receivers(G)
    best_n: dict[int, int] = {}
    best_z: dict[int, float] = {}
    for r in pairs.itertuples(index=False):
        for node in (r.gid_a, r.gid_b):
            if r.shared > best_n.get(node, 0):
                best_n[node] = r.shared
                best_z[node] = r.z
    df["shared_receivers_max"] = df.gid.map(best_n).fillna(0).astype(int)
    df["shared_receivers_z"] = df.gid.map(best_z).fillna(0.0).round(2)

    mids = shared_intermediaries(G)
    motifs = pd.concat([pairs, mids], ignore_index=True)
    return df, motifs


# ------------------------------------------------------- 1. возвратные потоки

def time_respecting_cycles(G: nx.DiGraph, tx: pd.DataFrame) -> set[int]:
    """Узлы из циклов, где даты переводов идут по возрастанию.

    Кольцо в графе само по себе не значит возврата средств: перевод A->B
    двадцатого числа и B->A пятого — две независимые операции. Проверяем,
    что по циклу существует последовательность транзакций с неубывающими датами.
    """
    dates: dict[tuple[int, int], list] = {}
    for r in tx.itertuples(index=False):
        dates.setdefault((r.src, r.dst), []).append(r.date)
    for k in dates:
        dates[k].sort()

    good: set[int] = set()
    for cycle in nx.simple_cycles(G, length_bound=CYCLE_MAX_LEN):
        hops = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
        cur = None
        ok = True
        for hop in hops:
            later = [d for d in dates.get(hop, []) if cur is None or d >= cur]
            if not later:
                ok = False
                break
            cur = later[0]
        if ok:
            good.update(cycle)
    return good


# --------------------------------------------------- 2. pass-through по платежам

def passthrough_matches(tx: pd.DataFrame) -> dict[int, int]:
    """Сколько входящих транзакций узла сопоставлено с исходящими.

    Сопоставление жадное и один-к-одному: каждая исходящая используется не более
    раза. Условие — сумма отличается не более чем на 3%, исходящая не раньше
    входящей и не позже чем через 2 дня.

    Это принципиально строже агрегатного transit_ratio: не «баланс похож»,
    а конкретная пара платежей, которую можно показать в выписке.
    """
    inc: dict[int, list] = defaultdict(list)
    out: dict[int, list] = defaultdict(list)
    for r in tx.itertuples(index=False):
        inc[r.dst].append((r.date, r.sum_kzt))
        out[r.src].append((r.date, r.sum_kzt))

    result: dict[int, int] = {}
    for node in set(inc) & set(out):
        outs = sorted(out[node])
        used = set()
        hits = 0
        for d_in, s_in in sorted(inc[node]):
            for i, (d_out, s_out) in enumerate(outs):
                if i in used or d_out < d_in:
                    continue
                if (d_out - d_in).days > DELAY_DAYS:
                    break
                if abs(s_out - s_in) <= AMOUNT_TOLERANCE * s_in:
                    used.add(i)
                    hits += 1
                    break
        if hits:
            result[node] = hits
    return result


# ------------------------------------------------- 3. общий пул получателей

def shared_receivers(G: nx.DiGraph) -> pd.DataFrame:
    """Пары отправителей с общими получателями, со значимостью по конфигурационной модели.

    Само пересечение ничего не доказывает: чем больше у отправителя получателей,
    тем выше шанс пересечься случайно. Поэтому для каждой пары считаем, сколько
    общих получателей дала бы случайная рассылка при ТЕХ ЖЕ степенях — вероятность
    выбрать получателя пропорциональна его входящей степени.
    """
    senders = {n: set(G.successors(n)) for n in G.nodes if G.out_degree(n) >= MIN_OUT_DEG_PAIR}
    if len(senders) < 2:
        return _empty_pairs()

    # инвертированный индекс: получатель -> отправители. Так пары находятся
    # без перебора всех сочетаний отправителей.
    by_receiver: dict[int, list[int]] = defaultdict(list)
    for s, rec in senders.items():
        for r in rec:
            by_receiver[r].append(s)

    overlap: dict[tuple[int, int], int] = defaultdict(int)
    for owners in by_receiver.values():
        for i in range(len(owners)):
            for j in range(i + 1, len(owners)):
                a, b = sorted((owners[i], owners[j]))
                overlap[(a, b)] += 1
    candidates = {k: v for k, v in overlap.items() if v >= MIN_OVERLAP}
    if not candidates:
        return _empty_pairs()

    in_deg = pd.Series(dict(G.in_degree()))
    in_deg = in_deg[in_deg > 0]
    probs = (in_deg / in_deg.sum()).values
    n_receivers = len(probs)

    rows = []
    cache: dict[tuple[int, int], tuple[float, float]] = {}
    for (a, b), shared in sorted(candidates.items(), key=lambda x: -x[1]):
        da, db = len(senders[a]), len(senders[b])
        key = (da, db)
        if key not in cache:
            sims = np.empty(SIMULATIONS)
            for k in range(SIMULATIONS):
                sa = RNG.choice(n_receivers, size=min(da, n_receivers), replace=False, p=probs)
                sb = RNG.choice(n_receivers, size=min(db, n_receivers), replace=False, p=probs)
                sims[k] = len(np.intersect1d(sa, sb, assume_unique=True))
            cache[key] = (float(sims.mean()), float(sims.std()) or 1e-9)
        exp, sd = cache[key]
        if exp < MIN_EXPECTED:
            continue
        rows.append({
            "kind": "shared_receivers",
            "gid_a": a,
            "gid_b": b,
            "shared": shared,
            "out_deg_a": da,
            "out_deg_b": db,
            "expected": round(exp, 2),
            "z": round((shared - exp) / sd, 2),
        })
    return pd.DataFrame(rows).sort_values("z", ascending=False).reset_index(drop=True)


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(columns=["kind", "gid_a", "gid_b", "shared",
                                 "out_deg_a", "out_deg_b", "expected", "z"])


# ----------------------------------------------- 4. общие посредники (scatter-gather)

def shared_intermediaries(G: nx.DiGraph, min_mid: int = 3) -> pd.DataFrame:
    """Пары A->[посредники]->B без прямого ребра A->B.

    Классический layering: прямой связи нет, поэтому анализ рёбер её не видит.
    Хабы исключаем из числа посредников: они и так фигурируют как концы пар,
    и без этого вывод получается круговым — узел объявляется одновременно
    и звеном схемы, и её организатором.
    """
    hubs = {n for n in G.nodes if G.out_degree(n) >= 20 or G.in_degree(n) >= 10}
    via: dict[tuple[int, int], set[int]] = defaultdict(set)
    for mid in G.nodes:
        if mid in hubs:
            continue
        for a in G.predecessors(mid):
            for b in G.successors(mid):
                if a != b:
                    via[(a, b)].add(mid)

    rows = [
        {"kind": "shared_intermediaries", "gid_a": a, "gid_b": b, "shared": len(m),
         "out_deg_a": G.out_degree(a), "out_deg_b": G.out_degree(b),
         "expected": None, "z": None}
        for (a, b), m in via.items()
        if len(m) >= min_mid and not G.has_edge(a, b)
    ]
    if not rows:
        return _empty_pairs()
    return pd.DataFrame(rows).sort_values("shared", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------- дописка в evidence

def augment_evidence(df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    """Дописывает найденные мотивы в конец evidence, не трогая исходный текст.

    Лимит ТЗ — 200 символов, его проверяет валидатор, поэтому обрезаем жёстко.
    """
    df = df.copy()
    extra = []
    for r in df.itertuples(index=False):
        bits = []
        if r.passthrough_matches >= 2:
            k = r.passthrough_matches
            bits.append(f"{k} {_plural(k, 'платёж прошёл', 'платежа прошли', 'платежей прошли')} насквозь")
        if r.cycle_time_ok:
            bits.append("деньги вернулись отправителю")
        if r.shared_receivers_z >= 3:
            bits.append(f"{r.shared_receivers_max} общих получателей (z={r.shared_receivers_z:.1f})")
        extra.append(("; " + "; ".join(bits)) if bits else "")
    df["evidence"] = [(e + x)[:limit] for e, x in zip(df.evidence, extra)]
    return df


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Русское склонение: «3 платежей прошли» читается как черновик."""
    n = abs(n) % 100
    if 11 <= n <= 14:
        return many
    n %= 10
    if n == 1:
        return one
    if 2 <= n <= 4:
        return few
    return many
