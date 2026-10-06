"""Carga de las series de precio: zonas de carga y, de antes, nodos sueltos."""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1] / "data"
ZONES_DIR = BASE / "zones"
NODES_DIR = BASE / "nodes"
PAPER_ZONE = "san_juan_del_rio"


def available_zones() -> list:
    """Zonas de carga disponibles en el panel."""
    return sorted(p.name[: -len(".csv.gz")] for p in ZONES_DIR.glob("*.csv.gz"))


def load_zone(zone: str = PAPER_ZONE) -> pd.DataFrame:
    """
    Las cuatro series de la zona: precio y sus tres componentes.

    El índice está en tiempo absoluto, no en hora local: así los días de 23 y 25
    horas del horario de verano —que México tuvo hasta finales de 2022— quedan sin
    ambigüedad y sin colisiones.
    """
    return pd.read_csv(ZONES_DIR / f"{zone}.csv.gz",
                       parse_dates=["ds"]).set_index("ds")


def load_series(column: str = "pml", zone: str = PAPER_ZONE,
                start: str = None, end: str = None) -> pd.Series:
    """Serie horaria de una zona, recortada al tramo pedido."""
    serie = load_zone(zone)[column].astype(float).dropna()
    if start or end:
        serie = serie.loc[start:end]
    return serie


def load_panel(zones: list, column: str = "pml",
               start: str = None, end: str = None) -> pd.DataFrame:
    """Una columna por zona, alineadas por hora."""
    datos = {z: load_series(column, z, start, end) for z in zones}
    return pd.DataFrame(datos)


## ---------------------------------------------------------------------------
## Los nodos del experimento anterior, que el artículo todavía cita
## ---------------------------------------------------------------------------

def available_nodes() -> list:
    return sorted(p.name[: -len(".csv.gz")] for p in NODES_DIR.glob("*.csv.gz"))


def load_node_series(column: str = "pml_mda", node: str = PAPER_ZONE) -> pd.Series:
    """
    Serie de un nodo del panel anterior, con los huecos de publicación rellenos.

    Se conserva porque los resultados publicados del artículo salieron de aquí.
    """
    tabla = pd.read_csv(NODES_DIR / f"{node}.csv.gz", parse_dates=["ds"])
    serie = tabla.set_index("ds")[column].astype(float).dropna()
    horas = pd.date_range(serie.index.min(), serie.index.max(), freq="h")
    return serie.reindex(horas).interpolate(limit_direction="both")
