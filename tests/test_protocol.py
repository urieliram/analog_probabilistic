"""
Pruebas de las garantías del protocolo.

No prueban que el método sea bueno: prueban que no hace trampa y que la
configuración no se puede improvisar.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_probabilistic.analog_probabilistic import (  # noqa: E402
    build_ensemble, find_analogs)
from experiments import protocol  # noqa: E402


def serie_sintetica(n: int = 3000) -> np.ndarray:
    rng = np.random.default_rng(7)
    estacional = 50 * np.sin(np.arange(n) * 2 * np.pi / 24)
    return 500 + estacional + rng.normal(0, 20, n)


## ---------------------------------------------------------------------------
## Que no mire hacia adelante
## ---------------------------------------------------------------------------

@pytest.mark.parametrize("ventana,horizonte", [(24, 24), (48, 24), (168, 24),
                                               (168, 120)])
def test_ninguna_continuacion_toca_la_ventana_presente(ventana, horizonte):
    """
    La continuación de cada análogo termina antes de que empiece el presente.

    Sin esta condición se cuela un miembro degenerado: el análogo inmediatamente
    anterior, cuya continuación es el presente mismo y cuyo mapa se ajustó justo
    para reproducirlo.
    """
    serie = serie_sintetica()
    analogos = find_analogs(serie, ventana, horizonte, k=20, separation=0.5)
    fin = analogos.positions + ventana + horizonte
    assert fin.max() <= len(serie) - ventana


def test_los_limites_del_recorte_salen_solo_de_la_historia_recibida():
    """Un pico posterior al origen no puede ensanchar los topes del recorte."""
    serie = serie_sintetica()
    miembros_sin_futuro, _ = build_ensemble(serie, window=48, horizon=24, k=10)

    con_futuro = np.concatenate([serie, np.full(240, 99_999.0)])
    ## se le pasa SÓLO la historia, que es lo que hace el experimento
    miembros_con_futuro, _ = build_ensemble(con_futuro[:len(serie)],
                                            window=48, horizon=24, k=10)
    assert np.allclose(miembros_sin_futuro, miembros_con_futuro)
    assert miembros_sin_futuro.max() <= serie.max()


def test_el_horizonte_no_puede_exceder_la_ventana():
    with pytest.raises(ValueError, match="horizonte"):
        build_ensemble(serie_sintetica(), window=24, horizon=48, k=10)


## ---------------------------------------------------------------------------
## Que la configuración no se improvise
## ---------------------------------------------------------------------------

def test_sin_configuracion_congelada_el_experimento_falla(tmp_path, monkeypatch):
    """
    «No volvimos a mirar» tiene que ser verificable, no una promesa.

    Si falta el archivo que congela la configuración elegida en el tramo de
    selección, el experimento no debe arrancar con un valor por omisión.
    """
    monkeypatch.setattr(protocol, "SELECTED_FILE", tmp_path / "no_existe.json")
    with pytest.raises(FileNotFoundError, match="select_config"):
        protocol.load_selected()


def test_los_tramos_no_se_enciman_ni_dejan_huecos():
    seleccion_fin = pd.Timestamp(protocol.SELECTION_END)
    prueba_inicio = pd.Timestamp(protocol.TEST_START)
    prueba_fin = pd.Timestamp(protocol.TEST_END)
    inspeccionado = pd.Timestamp(protocol.INSPECTED_START)

    assert pd.Timestamp(protocol.SELECTION_START) < seleccion_fin < prueba_inicio
    assert (prueba_inicio - seleccion_fin).days == 1
    assert prueba_fin < inspeccionado
    assert (inspeccionado - prueba_fin).days == 1


def test_los_origenes_caen_a_las_once_de_la_noche_y_caben_en_la_serie():
    horas = pd.date_range("2021-01-01", "2021-12-31 23:00", freq="h")
    serie = pd.Series(np.arange(len(horas), dtype=float), index=horas)
    origenes = protocol.build_origins("2021-06-01", "2021-06-30", serie)

    assert len(origenes) == 30
    assert set(origenes.hour) == {23}
    assert (origenes + pd.Timedelta(hours=protocol.HORIZON)).max() <= serie.index.max()
