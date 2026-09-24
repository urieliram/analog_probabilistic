"""Carga de las series de precio publicadas en data/nodes."""

from pathlib import Path

import pandas as pd

NODES_DIR = Path(__file__).resolve().parents[1] / "data" / "nodes"
PAPER_NODE = "san_juan_del_rio"


def available_nodes() -> list:
    """Nombres de los nodos que trae el repositorio."""
    return sorted(path.name[:-len(".csv.gz")] for path in NODES_DIR.glob("*.csv.gz"))


def load_node(node: str = PAPER_NODE) -> pd.DataFrame:
    """Las ocho series del nodo, tal como se publicaron."""
    return pd.read_csv(NODES_DIR / f"{node}.csv.gz", parse_dates=["ds"]).set_index("ds")


def load_series(column: str = "pml_mda", node: str = PAPER_NODE) -> pd.Series:
    """
    Devuelve la serie pedida, horaria y sin huecos.

    Las series del mercado en tiempo real terminan antes que las del día en
    adelanto porque el operador las publica con retraso, y el día en adelanto
    tiene algunas horas ausentes por fallas de publicación: se rellenan por
    interpolación lineal.
    """
    series = load_node(node)[column].astype(float).dropna()
    hours = pd.date_range(series.index.min(), series.index.max(), freq="h")
    return series.reindex(hours).interpolate(limit_direction="both")
