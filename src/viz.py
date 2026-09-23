"""Раскладка графа и автономный фолбэк на pyvis. Владелец: olzhasraiganiyev04-sketch.

    python -m src.viz        ->  out/graph.html

`out/graph.html` — самодостаточный файл: открывается двойным кликом **без сервера
и без интернета**. Это страховка на демо: если `app.py` не поднимется (занят порт,
не встала зависимость — что угодно), must-have №5 «схема сети» всё равно закрыт.

Здесь же живут функции раскладки: их импортирует `app.py`, чтобы веб-интерфейс
и фолбэк показывали граф одинаково. Координаты считаются на сервере и приходят
готовыми — браузер ничего не раскладывает, поэтому картинка появляется мгновенно
и не зависит от того, какие раскладки доступны в Cytoscape.
"""

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "out"

# Цвета ролей. Должны совпадать с ROLE_COLORS в web/index.html.
ROLE_COLORS = {
    "coordinator": "#f43f5e",
    "consolidator": "#F59E0B",
    "distributor": "#2DD4BF",
    "transit": "#818cf8",
    "terminal": "#64748b",
    "terminal_unknown": "#3f4a5c",
    "peripheral": "#2b3a52",
}

# Ширина символа подписи в координатах раскладки: шрифт ориентиров — 44,
# средний символ занимает примерно половину кегля. Нужна, чтобы прикинуть,
# влезает ли подпись в свой остров.
GUIDE_CHAR_W = 23.0

ROLE_RU = {
    "coordinator": "координатор",
    "consolidator": "накопитель",
    "distributor": "распределитель",
    "transit": "транзит",
    "terminal": "тупик",
    "terminal_unknown": "тупик не подтверждён",
    "peripheral": "периферия",
}


# --------------------------------------------------------------------------- #
#  Раскладки
# --------------------------------------------------------------------------- #

def flow_layout(nodes: list[dict], edges: list[dict],
                width: float = 2000, height: float = 1300) -> tuple[dict, list[dict]]:
    """«Поток»: x — колено обхода, y — барицентр соседей.

    Слева фигуранты (колено 0), дальше каждое следующее колено. Деньги на схеме
    текут слева направо, поэтому направление потока видно без чтения стрелок —
    это прямое требование ТЗ к экрану со схемой сети.

    Порядок внутри колонки — барицентрический (среднее по соседям), классическая
    эвристика Сугиямы: она заметно снижает число пересечений рёбер.
    Длинные колонки заворачиваются в несколько подколонок, иначе 171 узел
    вытягивается в нечитаемую нитку.

    Возвращает (координаты, подписи колонок).
    """
    if not nodes:
        return {}, []

    nbrs: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        if e["source"] in nbrs and e["target"] in nbrs:
            nbrs[e["source"]].append(e["target"])
            nbrs[e["target"]].append(e["source"])

    bands: dict[int, list[dict]] = {}
    for n in nodes:
        bands.setdefault(int(n.get("depth", 0)), []).append(n)
    keys = sorted(bands)

    band_x = {d: (width * i / max(1, len(keys) - 1)) for i, d in enumerate(keys)}
    band_gap = width / max(1, len(keys) - 1)

    # стартовый порядок: рядом стоят узлы одного кластера, крупные сверху
    order = {d: sorted(v, key=lambda n: (n.get("cluster", 0), -n.get("priority", 0)))
             for d, v in bands.items()}
    pos: dict[str, dict] = {}

    def place() -> None:
        for d in keys:
            col = order[d]
            rows = max(1, min(len(col), int(height // 34)))
            k = math.ceil(len(col) / rows)
            rows = math.ceil(len(col) / k)
            sub_gap = min(105.0, band_gap * 0.40 / max(1, k - 1)) if k > 1 else 0.0
            step = height / max(1, rows - 1) if rows > 1 else 0.0
            for j, n in enumerate(col):
                c, r = j // rows, j % rows
                pos[n["id"]] = {
                    "x": band_x[d] + (c - (k - 1) / 2) * sub_gap,
                    "y": (r - (rows - 1) / 2) * step,
                }

    place()
    # три прохода барицентра: каждый узел подтягивается к среднему уровню соседей
    for _ in range(3):
        for d in keys:
            order[d].sort(key=lambda n: (
                sum(pos[m]["y"] for m in nbrs[n["id"]] if m in pos) / len(nbrs[n["id"]])
                if nbrs[n["id"]] else pos[n["id"]]["y"]
            ))
        place()

    guides = [{
        "x": band_x[d], "y": -height / 2 - 80,
        "text": ("колено 0 · фигуранты" if d == 0 else f"колено {d}") + f" · {len(bands[d])}",
    } for d in keys]
    return pos, guides


def cluster_layout(nodes: list[dict], edges: list[dict],
                   width: float = 2000, height: float = 1300) -> tuple[dict, list[dict]]:
    """«Кластеры»: каждый кластер — отдельный остров, подписанный номером.

    Обычная силовая раскладка на этом графе не работает: 9 несвязных кусков
    разлетаются по углам, а главная компонента схлопывается в ком. Поэтому
    раскладываем каждый кластер отдельно и раскладываем острова полосами.
    Заодно это прямо показывает результат кластеризации — must-have №4.
    """
    import networkx as nx

    if not nodes:
        return {}, []

    groups: dict[int, list[dict]] = {}
    for n in nodes:
        groups.setdefault(int(n.get("cluster", 0)), []).append(n)
    ids_of = {c: {n["id"] for n in v} for c, v in groups.items()}

    # крупные и «дорогие» кластеры — первыми, то есть слева сверху
    order = sorted(groups, key=lambda c: (-len(groups[c]),
                                          -max(n.get("priority", 0) for n in groups[c])))

    pos: dict[str, dict] = {}
    guides: list[dict] = []
    pad, x, y, row_h, max_w = 70.0, 0.0, 0.0, 0.0, width * 1.15

    for c in order:
        members = groups[c]
        r = 26 * math.sqrt(len(members)) + 26          # радиус острова ~ размеру
        sub = nx.Graph()
        sub.add_nodes_from(ids_of[c])
        for e in edges:
            if e["source"] in ids_of[c] and e["target"] in ids_of[c]:
                sub.add_edge(e["source"], e["target"])

        if len(members) == 1:
            local = {members[0]["id"]: (0.0, 0.0)}
        else:
            local = nx.spring_layout(sub, seed=3, iterations=90,
                                     k=2.4 / math.sqrt(len(members)))
        mx = max((max(abs(p[0]), abs(p[1])) for p in local.values()), default=1.0) or 1.0

        if x + 2 * r + pad > max_w and x > 0:          # полка кончилась — переносим строку
            x, y = 0.0, y + row_h + pad
            row_h = 0.0
        cx, cy = x + r, y + r
        for nid, p in local.items():
            pos[nid] = {"x": cx + p[0] / mx * r, "y": cy + p[1] / mx * r}
        # Подпись не должна быть шире своего острова — иначе соседние сливаются
        # в нечитаемую строку. Не поместилась даже короткая: остров без подписи,
        # номер кластера всё равно виден в карточке узла.
        for text in (f"кластер {c} · {len(members)}", f"№{c} · {len(members)}"):
            if 2 * r > len(text) * GUIDE_CHAR_W:
                guides.append({"x": cx, "y": cy - r - 26, "text": text})
                break
        x += 2 * r + pad
        row_h = max(row_h, 2 * r)

    # центрируем всё поле относительно нуля — так удобнее и fit, и подписи
    xs = [p["x"] for p in pos.values()] or [0.0]
    ys = [p["y"] for p in pos.values()] or [0.0]
    dx, dy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    for p in pos.values():
        p["x"] -= dx
        p["y"] -= dy
    for g in guides:
        g["x"] -= dx
        g["y"] -= dy
    return pos, guides


LAYOUTS = {"flow": flow_layout, "clusters": cluster_layout}



# --------------------------------------------------------------------------- #
#  Автономный фолбэк
# --------------------------------------------------------------------------- #

def _load(out: Path) -> tuple[dict, dict]:
    """graph.json + карточки узлов из nodes_roles.csv (для подсказок)."""
    import pandas as pd

    graph = json.loads((out / "graph.json").read_text(encoding="utf-8"))
    df = pd.read_csv(out / "nodes_roles.csv")
    cards = {str(r.gid): r._asdict() for r in df.itertuples(index=False)}
    return graph, cards


def _tooltip(node: dict, card: dict) -> str:
    """Подсказка при наведении: gid, роль, обоснование, ключевые числа."""
    role = node.get("role", "—")
    lines = [f"gid {node['id']}",
             f"роль: {ROLE_RU.get(role, role)}",
             f"приоритет: {node.get('priority', 0):.2f}"]
    if card:
        if card.get("evidence"):
            lines.append(str(card["evidence"]))
        lines.append("получил {:,.0f} / отправил {:,.0f} KZT"
                     .format(card.get("in_kzt", 0) or 0, card.get("out_kzt", 0) or 0)
                     .replace(",", " "))
        lines.append("контрагентов: {} вх / {} исх"
                     .format(card.get("in_deg", 0), card.get("out_deg", 0)))
    if node.get("is_seed"):
        lines.append("фигурант (seed)")
    if role == "terminal_unknown":
        lines.append("4-е колено: конечность НЕ подтверждена")
    return "\n".join(lines)


def build(out: Path = OUT, layout: str = "flow", label_top: int = 12) -> Path:
    """Пишет `out/graph.html`. Возвращает путь к файлу."""
    from pyvis.network import Network

    graph, cards = _load(out)
    nodes, edges = graph["nodes"], graph["edges"]
    pos, guides = LAYOUTS.get(layout, flow_layout)(nodes, edges)

    # cdn_resources='in_line' вместо 'remote': фолбэк нужен ровно в тот момент,
    # когда всё остальное сломалось, — полагаться на интернет в зале нельзя.
    net = Network(height="100vh", width="100%", directed=True,
                  bgcolor="#0B1E3A", font_color="#e8eef7",
                  notebook=False, cdn_resources="in_line")
    net.toggle_physics(False)          # координаты уже посчитаны — физика только испортит

    # подписи колонок: без них колена обхода приходится объяснять словами
    for i, g in enumerate(guides):
        net.add_node(f"__guide{i}", label=g["text"], x=g["x"], y=g["y"], physics=False,
                     shape="text")
    # `font` в add_node pyvis молча отбрасывает (проверено: ключа нет в готовом HTML),
    # поэтому дописываем его прямо в узел — иначе подписи остаются кеглем 14.
    for node in net.nodes:
        if isinstance(node, dict) and str(node.get("id", "")).startswith("__guide"):
            node["font"] = {"size": 40, "color": "#8fa7c9"}

    named = {n["id"] for n in sorted(nodes, key=lambda n: -n.get("priority", 0))[:label_top]}
    for n in nodes:
        p = pos.get(n["id"], {"x": 0, "y": 0})
        net.add_node(
            n["id"],
            label=n["id"][-6:] if n["id"] in named else " ",
            title=_tooltip(n, cards.get(n["id"], {})),
            color=ROLE_COLORS.get(n.get("role"), "#2b3a52"),
            size=6 + n.get("priority", 0) * 26,
            borderWidth=3 if n.get("is_seed") else 1,
            x=p["x"], y=p["y"], physics=False,
        )

    mx = max((float(e.get("sum_kzt", 0)) for e in edges), default=1.0) or 1.0
    ids = {n["id"] for n in nodes}
    for e in edges:
        if e["source"] not in ids or e["target"] not in ids:
            continue
        s = float(e.get("sum_kzt", 0))
        net.add_edge(e["source"], e["target"],
                     width=0.6 + 5.4 * (s / mx) ** 0.4,
                     color="#2f5a94",
                     title="{:,.0f} KZT, {} транзакций".format(s, e.get("n_tx", 0)).replace(",", " "))

    net.set_options(json.dumps({
        "interaction": {"hover": True, "tooltipDelay": 80},
        "edges": {"arrows": {"to": {"enabled": True, "scaleFactor": 0.5}}, "smooth": False},
        "physics": {"enabled": False},
    }))

    out.mkdir(parents=True, exist_ok=True)
    path = out / "graph.html"
    net.write_html(str(path), notebook=False, open_browser=False)
    _add_legend(path, nodes)
    return path


def _add_legend(path: Path, nodes: list[dict]) -> None:
    """Легенда ролей поверх схемы — в pyvis её нет, дописываем в готовый HTML."""
    counts: dict[str, int] = {}
    for n in nodes:
        counts[n.get("role", "—")] = counts.get(n.get("role", "—"), 0) + 1
    items = "".join(
        f'<span><i style="background:{ROLE_COLORS.get(r, "#2b3a52")}"></i>'
        f'{ROLE_RU.get(r, r)} — {c}</span>'
        for r, c in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    # Легенда — отдельной полосой сверху, а не плашкой поверх схемы:
    # наложение съедало левую колонку графа после подгонки масштаба.
    block = f"""
<style>
 body{{background:#0B1E3A;margin:0;
   font:13px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#e8eef7}}
 #legend{{padding:10px 16px;background:#11294c;border-bottom:1px solid #1d3a63}}
 #legend .t{{font-size:15px;font-weight:700;margin-right:18px}}
 #legend .roles{{display:inline-flex;flex-wrap:wrap;gap:6px 16px;vertical-align:middle}}
 #legend .roles span{{white-space:nowrap}}
 #legend i{{display:inline-block;width:10px;height:10px;border-radius:99px;margin-right:6px}}
 #legend .n{{color:#93a8c6;font-size:11.5px;margin-top:5px}}
 #mynetwork{{height:calc(100vh - 82px) !important;border:0 !important}}
 .card{{border:0 !important;background:transparent !important}}
</style>
<div id="legend">
 <span class="t">Граф денег — автономная схема</span><span class="roles">{items}</span>
 <div class="n">Размер узла — приоритет проверки, толщина ребра — сумма перевода,
 стрелка — направление денег. Наведите курсор на узел, чтобы увидеть обоснование роли.
 Показаны {len(nodes)} узлов с наибольшим приоритетом и их окружение; остальные ищутся
 по gid в веб-интерфейсе (<code>python app.py</code>).</div>
</div>
<script>
 // Координаты заданы жёстко, физика выключена — vis сам вид не подгоняет,
 // и граф открывается вплотную. Подгоняем масштаб после отрисовки.
 window.addEventListener('load', function () {{
   setTimeout(function () {{
     try {{ network.fit({{animation: false}}); }} catch (e) {{}}
   }}, 120);
 }});
</script>
"""
    html = path.read_text(encoding="utf-8")
    path.write_text(html.replace("<body>", "<body>" + block, 1), encoding="utf-8")


def main() -> None:
    if not (OUT / "graph.json").exists():
        raise SystemExit("Нет out/graph.json. Сначала: python run.py  (или python -m src.mock)")
    print(f"автономная схема записана: {build()}")


if __name__ == "__main__":
    main()
