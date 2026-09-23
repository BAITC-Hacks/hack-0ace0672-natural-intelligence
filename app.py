#!/usr/bin/env python3
"""Веб-интерфейс аналитика. Владелец: olzhasraiganiyev04-sketch.

    python app.py     ->  http://localhost:8000

Ключ DeepSeek остаётся здесь, на бэке, и в браузер не попадает.
Данные берутся из out/ — сначала запустите `python run.py`
(или `python -m src.mock`, если ядро ещё не готово).
"""

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
WEB = ROOT / "web"

app = FastAPI(title="Граф денег")


def _nodes() -> pd.DataFrame:
    if not (OUT / "nodes_roles.csv").exists():
        raise HTTPException(503, "Нет выгрузок. Запустите: python run.py")
    return pd.read_csv(OUT / "nodes_roles.csv")


@app.get("/api/graph")
def graph():
    p = OUT / "graph.json"
    if not p.exists():
        raise HTTPException(503, "Нет graph.json. Запустите: python run.py")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/node/{gid}")
def node(gid: int):
    from src import tools
    data = tools.get_node(gid)
    if "error" in data:
        raise HTTPException(404, data["error"])
    return json.loads(json.dumps(data, default=str))


@app.get("/api/top")
def top():
    return pd.read_csv(OUT / "top_nodes.csv").to_dict("records")


@app.get("/api/clusters")
def clusters():
    return pd.read_csv(OUT / "clusters.csv").to_dict("records")


@app.get("/api/roles")
def roles():
    return _nodes().role.value_counts().to_dict()


class Ask(BaseModel):
    question: str


@app.post("/api/ask")
def ask(body: Ask):
    from src import agent
    return agent.answer(body.question)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/web", StaticFiles(directory=WEB), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
