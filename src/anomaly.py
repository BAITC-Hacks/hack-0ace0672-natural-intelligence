"""Аномальные профили узлов относительно их колена. Владелец: kuanyshs.

Опциональный пункт ТЗ формулирует требование точно: аномалия считается
ОТНОСИТЕЛЬНО КОЛЕНА. Узел depth=1 с 60 получателями и узел depth=4 с 60
получателями — разные истории: на первом колене это норма, на четвёртом выброс.
Глобальная модель их смешает.

Почему не IsolationForest. Он выдаёт число, и на вопрос «почему 0.87» ответить
нечем — прямое попадание в запрет ТЗ на чёрный ящик. Забор по межквартильному
размаху внутри колена даёт то же самое и ещё называет, КАКОЙ признак выбивается
и во сколько раз. Это заменяет SHAP и считается мгновенно.

Почему не z-скор. Признаки здесь дискретные и малые: медиана out_deg равна 2,
у половины узлов колена значения совпадают, поэтому MAD и дисперсия близки к нулю
и обычная вариация превращается в 50 сигм. На первой версии так и вышло — порог
z>=3 отмечал 799 узлов, то есть треть графа. Забор Q3 + 3*IQR — стандартное
определение дальнего выброса, оно от этой проблемы свободно.
"""

import pandas as pd

from src.text import kzt, plural

# Признаки, по которым ищем выброс. Только те, что аналитик понимает без пояснений.
FEATURES = {
    "in_deg": "плательщиков",
    "out_deg": "получателей",
    "in_kzt": "входящий оборот",
    "out_kzt": "исходящий оборот",
    "external_funding_gap": "приток извне выборки",
    "max_edge_n_tx": "переводов по одному направлению",
    "n_seeds_upstream": "фигурантов выше по течению",
}

IQR_K = 3.0          # множитель для ДАЛЬНЕГО выброса (обычный выброс — 1.5)
TAIL_QUANTILE = 0.99 # запасная граница, когда межквартильный размах равен нулю
MIN_FLAGS = 1      # с одного выбивающегося признака считаем профиль аномальным


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет anomaly_flags, anomaly_ratio и anomaly_why. Роли не трогает."""
    df = df.copy()
    exceed = pd.DataFrame(index=df.index)   # во сколько раз превышен забор
    ratio = pd.DataFrame(index=df.index)    # во сколько раз больше медианы колена

    for col in FEATURES:
        if col not in df.columns:
            continue
        ex = pd.Series(0.0, index=df.index)
        bound = pd.Series(0.0, index=df.index)
        for _, idx in df.groupby("depth").groups.items():
            v = df.loc[idx, col].astype(float)
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            # Если средняя половина колена не имеет разброса (q1 == q3 — обычное
            # дело для дискретных степеней), забор по IQR вырождается и отмечает
            # всех подряд. Тогда границей служит верхний процент колена.
            fence = max(q3 + IQR_K * (q3 - q1), v.quantile(TAIL_QUANTILE))
            ex.loc[idx] = (v / fence).where(v > fence, 0.0) if fence > 0 else 0.0
            bound.loc[idx] = fence
        exceed[col] = ex.fillna(0.0)
        ratio[col] = bound.fillna(0.0)

    df["anomaly_flags"] = (exceed > 0).sum(axis=1).astype(int)
    df["anomaly_ratio"] = exceed.max(axis=1).round(1)
    df["anomaly_why"] = _explain(exceed, ratio, df)
    return df


def _explain(exceed: pd.DataFrame, bound: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    """Называет два самых выбивающихся признака — это и есть объяснение аномалии.

    Сравниваем с ГРАНИЦЕЙ колена, а не с медианой: медиана оборота на втором
    колене равна нулю, и деление на неё давало «в 23 миллиона раз больше».
    """
    cols = list(exceed.columns)
    money = {"in_kzt", "out_kzt", "external_funding_gap"}
    order = exceed.values.argsort(axis=1)[:, ::-1][:, :2]
    out = []
    for pos, ((i, j), depth) in enumerate(zip(order, df.depth.values)):
        bits = []
        for k in (i, j):
            factor = exceed.values[pos, k]
            if factor <= 0:
                continue
            col = cols[k]
            value = df[col].values[pos]
            shown = kzt(value) if col in money else f"{value:.0f}"
            lim = kzt(bound.values[pos, k]) if col in money else f"{bound.values[pos, k]:.0f}"
            times = plural(int(factor), "раз", "раза", "раз")
            bits.append(f"{FEATURES[col]} {shown} — в {factor:.0f} {times} выше границы "
                        f"{depth}-го колена ({lim})")
        out.append("; ".join(bits) if bits else "профиль в пределах своего колена")
    return out
