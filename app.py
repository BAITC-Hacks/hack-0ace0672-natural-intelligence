#!/usr/bin/env python3
"""Веб-интерфейс аналитика. Владелец: olzhasraiganiyev04-sketch.

    python app.py     ->  http://localhost:8000

Ключ DeepSeek остаётся здесь, на бэке, и в браузер не попадает.
Данные берутся из out/ — сначала запустите `python run.py`
(или `python -m src.mock`, если ядро ещё не готово).

Два решения, которые стоит знать при чтении кода:

1. **gid всегда отдаётся строкой.** Это 18-значные числа, у double мантисса 53 бита,
   поэтому в JSON-числе gid теряет точность: 100000000343175100 приезжает в браузер
   как ...104. Строка снимает весь класс проблемы разом.
2. **Координаты узлов считает сервер** (`src/viz.py`), браузер их только рисует.
   Раскладка детерминирована, появляется мгновенно и одинакова в веб-интерфейсе
   и в автономном `out/graph.html`.
"""

import json
import threading
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import viz

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
WEB = ROOT / "web"
DATA = ROOT / "data"

app = FastAPI(title="Граф денег")

# Колонки карточки узла: всё, что показывает интерфейс. Лишнее в браузер не гоним.
CARD_COLUMNS = [
    "gid", "role", "role_score", "cluster_id", "priority_score", "evidence",
    "depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt",
    "transit_ratio", "net_flow", "external_funding_gap", "taint_share",
    "external_share", "median_delay_days", "n_seeds_upstream",
]


# --------------------------------------------------------------------------- #
#  Загрузка выгрузок. Кэш сбрасывается сам, если run.py перезаписал файлы.
# --------------------------------------------------------------------------- #

_cache: dict[str, tuple[float, object]] = {}
_layouts: dict[tuple[str, float], dict] = {}


def _cached(path: Path, loader):
    if not path.exists():
        raise HTTPException(503, f"Нет файла {path.name}. Запустите: python run.py")
    stamp = path.stat().st_mtime
    hit = _cache.get(str(path))
    if hit is None or hit[0] != stamp:
        _cache[str(path)] = (stamp, loader(path))
    return _cache[str(path)][1]


def _nodes() -> pd.DataFrame:
    """nodes_roles.csv. gid — строка: так он совпадает с id узлов в graph.json."""
    def load(p: Path) -> pd.DataFrame:
        df = pd.read_csv(p)
        df["gid"] = df.gid.astype(str)
        return df
    return _cached(OUT / "nodes_roles.csv", load)


def _edges() -> pd.DataFrame:
    def load(p: Path) -> pd.DataFrame:
        df = pd.read_parquet(p)[["src", "dst", "sum_kzt", "n_tx"]]
        df["src"] = df.src.astype(str)
        df["dst"] = df.dst.astype(str)
        return df
    return _cached(DATA / "edges.parquet", load)


def _graph_json() -> dict:
    return _cached(OUT / "graph.json", lambda p: json.loads(p.read_text(encoding="utf-8")))


def _clean(value):
    """NaN не является валидным JSON — в браузер уходит null."""
    if isinstance(value, float) and value != value:
        return None
    if hasattr(value, "item"):
        return _clean(value.item())
    return value


def _card(gid: str) -> dict:
    """Строка из nodes_roles.csv в виде словаря, gid строкой."""
    df = _nodes()
    row = df[df.gid == gid]
    if row.empty:
        raise HTTPException(404, f"gid {gid} нет в выгрузке")
    r = row.iloc[0]
    out = {c: _clean(r[c]) for c in CARD_COLUMNS if c in row.columns}
    out["gid"] = gid
    return out


# --------------------------------------------------------------------------- #
#  Схема сети
# --------------------------------------------------------------------------- #

@app.get("/api/graph")
def graph(layout: str = "flow"):
    """Узлы и рёбра для отрисовки, с готовыми координатами.

    `layout=flow`     — колонки по коленам обхода, деньги текут слева направо;
    `layout=clusters` — каждый кластер отдельным подписанным островом.
    """
    if layout not in viz.LAYOUTS:
        raise HTTPException(400, f"Неизвестная раскладка: {layout}")
    data = _graph_json()
    stamp = (OUT / "graph.json").stat().st_mtime
    if (layout, stamp) not in _layouts:
        # graph.json перезаписан — координаты для старой версии больше не нужны
        for k in [k for k in _layouts if k[1] != stamp]:
            del _layouts[k]
        _layouts[(layout, stamp)] = viz.LAYOUTS[layout](data["nodes"], data["edges"])
    pos, guides = _layouts[(layout, stamp)]

    nodes = [{**n, "x": pos.get(n["id"], {}).get("x", 0.0),
              "y": pos.get(n["id"], {}).get("y", 0.0)} for n in data["nodes"]]
    return {"nodes": nodes, "edges": data["edges"], "layout": layout, "guides": guides}


@app.get("/api/meta")
def meta():
    """Шапка интерфейса: сколько узлов всего и сколько из них на схеме."""
    df = _nodes()
    shown = {n["id"] for n in _graph_json()["nodes"]}
    e = _edges()
    return {
        "n_nodes": int(len(df)),
        "n_seeds": int(df.is_seed.sum()) if "is_seed" in df else 0,
        "n_edges": int(len(e)),
        "n_shown": len(shown),
        "turnover_kzt": float(e.sum_kzt.sum()),
        "roles": df.role.value_counts().to_dict(),
        "roles_shown": pd.Series([n["role"] for n in _graph_json()["nodes"]])
                         .value_counts().to_dict(),
    }


@app.get("/api/roles")
def roles():
    return _nodes().role.value_counts().to_dict()


# --------------------------------------------------------------------------- #
#  Узел и его окружение
# --------------------------------------------------------------------------- #

@app.get("/api/node/{gid}")
def node(gid: str):
    """Карточка узла: роль, обоснование, числа, крупнейшие контрагенты.

    gid принимаем строкой — целочисленный путь на 18 знаках уже приводил
    к расхождению между тем, что показано, и тем, что запрошено.
    """
    card = _card(gid)
    e = _edges()
    senders = e[e.dst == gid].nlargest(6, "sum_kzt")
    receivers = e[e.src == gid].nlargest(6, "sum_kzt")
    card["top_senders"] = [{"gid": r.src, "sum_kzt": float(r.sum_kzt), "n_tx": int(r.n_tx)}
                           for r in senders.itertuples(index=False)]
    card["top_receivers"] = [{"gid": r.dst, "sum_kzt": float(r.sum_kzt), "n_tx": int(r.n_tx)}
                             for r in receivers.itertuples(index=False)]
    card["on_screen"] = gid in {n["id"] for n in _graph_json()["nodes"]}
    return card


@app.get("/api/subgraph/{gid}")
def subgraph(gid: str, hops: int = 1, limit: int = 120):
    """Окружение узла отдельным запросом — для поиска по gid вне схемы.

    На схеме 500 узлов из 2248, и жюри вполне может назвать gid, которого там нет.
    Тогда фронт подгружает окружение этим роутом и дорисовывает его на месте,
    вместо того чтобы молча ничего не сделать.
    """
    _card(gid)                      # 404, если gid вообще нет в выгрузке
    e = _edges()
    keep = {gid}
    for _ in range(max(1, min(hops, 2))):
        near = e[e.src.isin(keep) | e.dst.isin(keep)]
        keep |= set(near.src) | set(near.dst)
        if len(keep) > limit:
            break

    # если узлов слишком много — оставляем центр и самых крупных контрагентов
    truncated = False
    if len(keep) > limit:
        truncated = True
        inc = e[(e.src == gid) | (e.dst == gid)].nlargest(limit - 1, "sum_kzt")
        keep = {gid} | set(inc.src) | set(inc.dst)

    df = _nodes()
    sub = df[df.gid.isin(keep)]
    sub_edges = e[e.src.isin(keep) & e.dst.isin(keep)]
    return {
        "center": gid,
        "truncated": truncated,
        "nodes": [{"id": r.gid, "role": r.role, "priority": float(r.priority_score),
                   "cluster": int(r.cluster_id), "depth": int(r.depth),
                   "is_seed": bool(r.is_seed)}
                  for r in sub.itertuples(index=False)],
        "edges": [{"source": r.src, "target": r.dst,
                   "sum_kzt": float(r.sum_kzt), "n_tx": int(r.n_tx)}
                  for r in sub_edges.itertuples(index=False)],
    }


# --------------------------------------------------------------------------- #
#  Списки
# --------------------------------------------------------------------------- #

@app.get("/api/top")
def top():
    df = _cached(OUT / "top_nodes.csv", pd.read_csv).copy()
    df["gid"] = df.gid.astype(str)
    return json.loads(df.to_json(orient="records"))


@app.get("/api/resilience")
def resilience():
    """Устойчивость сети при изъятии топ-N узлов (src/resilience.py, владелец kuanyshs).

    Файла может не быть — если пайплайн запускали старой версией. Тогда отдаём
    пустой список, а интерфейс просто не рисует блок.
    """
    p = OUT / "resilience.csv"
    if not p.exists():
        return []
    return json.loads(_cached(p, pd.read_csv).to_json(orient="records"))


@app.get("/api/clusters")
def clusters():
    df = _cached(OUT / "clusters.csv", pd.read_csv).copy()
    if "top_gids" in df.columns:
        df["top_gids"] = df.top_gids.astype(str)
    return json.loads(df.to_json(orient="records"))


# --------------------------------------------------------------------------- #
#  Ассистент
# --------------------------------------------------------------------------- #

class Ask(BaseModel):
    question: str


@app.post("/api/ask")
def ask(body: Ask):
    """Проксирует вопрос в src.agent. Файл чужой — падение агента гасим здесь,
    чтобы интерфейс на демо остался живым."""
    try:
        from src import agent
        res = agent.answer(body.question)
    except Exception as exc:                       # noqa: BLE001 — демо важнее стектрейса
        return {"answer": f"Ассистент недоступен: {exc}", "cited_gids": [], "mode": "error"}
    res["cited_gids"] = [str(g) for g in res.get("cited_gids", [])]
    return res


# --------------------------------------------------------------------------- #
#  Статика и автономная схема
# --------------------------------------------------------------------------- #

@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/fallback")
def fallback():
    """Автономная схема на pyvis. Тот же файл, что открывается двойным кликом."""
    p = OUT / "graph.html"
    if not p.exists():
        try:
            viz.build(OUT)
        except Exception as exc:                   # noqa: BLE001
            raise HTTPException(503, f"Не удалось собрать graph.html: {exc}")
    return FileResponse(p)


def _build_fallback() -> None:
    """Пересобирает out/graph.html при старте — фоном, старт сервера не задерживает."""
    try:
        viz.build(OUT)
        print("автономная схема готова: out/graph.html (открывается без сервера)")
    except Exception as exc:                       # noqa: BLE001
        print(f"автономная схема не собралась ({exc}) — интерфейс это не ломает")


app.mount("/web", StaticFiles(directory=WEB), name="web")


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Веб-интерфейс «Граф денег»")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()

    if (OUT / "graph.json").exists():
        threading.Thread(target=_build_fallback, daemon=True).start()
    else:
        print("Нет out/graph.json — сначала запустите: python run.py")

    print(f"интерфейс аналитика: http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port)
