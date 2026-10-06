"""
Que separar producir de calificar no cambie ningún número.

Es la prueba que vuelve seguro el cambio: si guardar los pronósticos y calificarlos
después diera algo distinto de calificarlos al vuelo, el refactor habría metido un
error silencioso en todos los cuadros a la vez.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate import evaluar_archivo  # noqa: E402
from experiments.forecast_store import (ForecastStore,  # noqa: E402
                                        load_members, member_columns,
                                        quantile_columns)
from experiments.scores import evaluate  # noqa: E402

NIVELES = [0.1, 0.25, 0.5, 0.75, 0.9]


def pronostico_sintetico(semilla: int = 3):
    rng = np.random.default_rng(semilla)
    observado = 500 + rng.normal(0, 50, 24)
    miembros = observado + rng.normal(0, 40, (12, 24))
    cuantiles = np.quantile(miembros, NIVELES, axis=0)
    return observado, miembros, cuantiles


def test_guardar_y_calificar_da_lo_mismo_que_calificar_al_vuelo(tmp_path):
    observado, miembros, cuantiles = pronostico_sintetico()
    al_vuelo = evaluate(observado, cuantiles, NIVELES, miembros)

    almacen = ForecastStore(NIVELES, tmp_path / "corrida.parquet")
    almacen.add("zona_a", pd.Timestamp("2024-05-01 23:00"), "analogo",
                cuantiles, observado, miembros, seconds=0.03)
    almacen.save()

    guardado = evaluar_archivo(tmp_path / "corrida.parquet", NIVELES).iloc[0]
    for puntaje, valor in al_vuelo.items():
        assert guardado[puntaje] == pytest.approx(valor), puntaje


def test_los_miembros_sobreviven_el_viaje(tmp_path):
    observado, miembros, cuantiles = pronostico_sintetico()
    almacen = ForecastStore(NIVELES, tmp_path / "corrida.parquet")
    almacen.add("zona_a", pd.Timestamp("2024-05-01 23:00"), "analogo",
                cuantiles, observado, miembros)
    almacen.save()

    recuperados = load_members(tmp_path / "corrida.parquet")
    columnas = member_columns(recuperados)
    assert np.allclose(recuperados.sort_values("step")[columnas].to_numpy().T,
                       miembros)


def test_califica_sobre_la_rejilla_pedida_y_no_sobre_la_que_haya(tmp_path):
    """
    Un método con muchos cuantiles no puede calificarse con su propia resolución.

    Si se evalúa sobre la rejilla común, el resultado tiene que ser idéntico al de
    calcular a mano sobre esos mismos niveles, aunque el archivo traiga más.
    """
    observado, miembros, _ = pronostico_sintetico()
    finos = [round(a, 2) for a in np.arange(0.05, 0.96, 0.05)]
    cuantiles = np.quantile(miembros, finos, axis=0)

    almacen = ForecastStore(finos, tmp_path / "corrida.parquet")
    almacen.add("zona_a", pd.Timestamp("2024-05-01 23:00"), "analogo",
                cuantiles, observado)
    almacen.save()

    comunes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    guardado = evaluar_archivo(tmp_path / "corrida.parquet", comunes).iloc[0]
    esperado = evaluate(observado, np.quantile(miembros, comunes, axis=0), comunes)
    assert guardado["mean_pinball"] == pytest.approx(esperado["mean_pinball"])


def test_el_almacen_rechaza_cuantiles_que_no_cuadran(tmp_path):
    observado, _, cuantiles = pronostico_sintetico()
    almacen = ForecastStore(NIVELES + [0.95], tmp_path / "corrida.parquet")
    with pytest.raises(ValueError, match="niveles"):
        almacen.add("zona_a", pd.Timestamp("2024-05-01 23:00"), "analogo",
                    cuantiles, observado)


def test_las_columnas_de_cuantil_salen_ordenadas(tmp_path):
    observado, _, cuantiles = pronostico_sintetico()
    almacen = ForecastStore(NIVELES, tmp_path / "corrida.parquet")
    almacen.add("z", pd.Timestamp("2024-05-01 23:00"), "m", cuantiles, observado)
    almacen.save()
    _, niveles = quantile_columns(pd.read_parquet(tmp_path / "corrida.parquet"))
    assert niveles == sorted(niveles)
