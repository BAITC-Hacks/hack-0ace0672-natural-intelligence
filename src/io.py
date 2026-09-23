"""Загрузка данных и сборка графа. Владелец: kuanyshs."""

from pathlib import Path

import networkx as nx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def load(data_dir: Path = DATA_DIR):
    """Три parquet-файла. tx['date'] приводится к datetime."""
    edges = pd.read_parquet(data_dir / "edges.parquet")
    nodes = pd.read_parquet(data_dir / "nodes.parquet")
    tx = pd.read_parquet(data_dir / "transactions.parquet")
    tx["date"] = pd.to_datetime(tx["date"])
    return edges, nodes, tx


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    """Направленный взвешенный граф. Вес ребра — sum_kzt, плюс n_tx и depth."""
    G = nx.DiGraph()
    for r in edges.itertuples(index=False):
        G.add_edge(r.src, r.dst, sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx), depth=int(r.depth))
    return G


def seeds(nodes: pd.DataFrame) -> list[int]:
    return nodes.loc[nodes.is_seed, "gid"].tolist()
