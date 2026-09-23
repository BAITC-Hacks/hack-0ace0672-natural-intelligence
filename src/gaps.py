"""Оценка полноты: каких данных не хватает и что запрашивать следующим.

Владелец: kuanyshs. Опциональный пункт ТЗ.

Смысл не в том, чтобы перечислить ограничения выгрузки в README — это мы и так
делаем. Смысл в том, чтобы для КАЖДОГО узла в топе сказать, какой запрос закроет
белое пятно именно по нему. Аналитик получает не список претензий к данным,
а следующее действие.

Правила упорядочены по ценности запроса: сначала то, что меняет картину сильнее.
"""

import pandas as pd

from src.text import kzt

GAP_MIN_KZT = 500_000      # ниже этого разрыв не стоит отдельного запроса
NEAR_THRESHOLD_TX = 5      # много переводов по одному ребру -> вероятно дробление


def next_request(r) -> str:
    """Какой запрос по этому узлу даст больше всего новой информации."""
    if getattr(r, "external_funding_gap", 0) >= GAP_MIN_KZT:
        return (f"запросить входящие переводы: {kzt(r.external_funding_gap)} "
                f"пришло из-за пределов выборки, происхождение неизвестно")
    if getattr(r, "role", "") == "terminal_unknown":
        return ("запросить исходящие переводы: обход оборван на 4-м колене, "
                "конечность узла не подтверждена")
    if getattr(r, "is_seed", False):
        return ("запросить входящие переводы: у фигурантов дела входящие суммы "
                "занижены по построению выгрузки")
    if getattr(r, "max_edge_n_tx", 0) >= NEAR_THRESHOLD_TX:
        return (f"запросить переводы ниже 5 000 KZT: до {int(r.max_edge_n_tx)} операций "
                f"по одному направлению, дробление ниже порога невидимо")
    if getattr(r, "in_deg", 0) == 0 and getattr(r, "out_deg", 0) > 0:
        return "запросить входящие переводы: источник средств узла в выгрузке отсутствует"
    return "дополнительных запросов по узлу не требуется"


def add_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["next_request"] = [next_request(r) for r in df.itertuples(index=False)]
    return df


def summary(df: pd.DataFrame) -> str:
    """Сводка для README и демо: сколько узлов требуют какого запроса."""
    counts = df.next_request.str.split(":").str[0].value_counts()
    lines = ["ОЦЕНКА ПОЛНОТЫ: какие запросы закроют белые пятна"]
    for name, k in counts.items():
        lines.append(f"  {k:>5}  {name}")

    # Seed исключаем: у фигурантов дела входящие занижены по построению выгрузки,
    # их разрыв — артефакт обхода, а не находка. С ними цифра раздувается
    # до 275 млн и расходится с тем, что написано в README.
    ext = df[(df.external_funding_gap > 0) & (~df.is_seed)]
    gap = float(ext.external_funding_gap.sum())
    share = gap / df.out_kzt.sum() * 100 if df.out_kzt.sum() else 0
    lines.append(f"\n  {len(ext)} не-seed узлов принесли {kzt(gap)} из-за пределов "
                 f"выборки ({share:.0f}% оборота) — это главный пробел")
    return "\n".join(lines)
