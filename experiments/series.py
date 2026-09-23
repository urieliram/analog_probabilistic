"""Carga de la serie de precios del nodo San Juan del Río."""

from pathlib import Path

import pandas as pd

DATA_FILE = (Path(__file__).resolve().parents[1] / "data"
             / "san_juan_del_rio_lmp_hourly.csv")


def load_series(column: str = "pml_mda") -> pd.Series:
    """
    Devuelve la serie horaria pedida, sin huecos.

    El archivo trae las ocho series del nodo. Las cuatro del mercado en tiempo
    real terminan antes que las del día en adelanto porque el operador las
    publica con retraso, y el día en adelanto tiene cuatro horas ausentes por
    fallas de publicación: se rellenan por interpolación lineal.
    """
    frame = pd.read_csv(DATA_FILE, parse_dates=["ds"]).set_index("ds")
    series = frame[column].astype(float).dropna()
    hours = pd.date_range(series.index.min(), series.index.max(), freq="h")
    return series.reindex(hours).interpolate(limit_direction="both")
