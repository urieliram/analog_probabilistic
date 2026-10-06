"""
Pruebas de los puntajes, con casos de solución conocida a mano.

La razón de que existan: el artículo llamó CRPS a la pérdida pinball media y llamó
intervalo de 95% al central de 90%. Ninguna de las dos cosas la habría detectado
una prueba de humo; sí la detecta un caso resuelto a mano.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.scores import (crps_ensemble, interval_score,  # noqa: E402
                                mean_pinball, pinball,
                                weighted_interval_score)


def test_pinball_castiga_mas_el_lado_que_le_toca():
    ## con nivel 0.9, quedarse corto cuesta nueve veces más que pasarse
    assert pinball(np.array([10.0]), np.array([0.0]), 0.9)[0] == pytest.approx(9.0)
    assert pinball(np.array([0.0]), np.array([10.0]), 0.9)[0] == pytest.approx(1.0)


def test_pinball_en_la_mediana_es_medio_error_absoluto():
    observado, cuantil = np.array([7.0]), np.array([3.0])
    assert pinball(observado, cuantil, 0.5)[0] == pytest.approx(2.0)


def test_mean_pinball_promedia_niveles_y_horizonte():
    observado = np.array([1.0, 1.0])
    cuantiles = np.array([[0.0, 0.0], [2.0, 2.0]])
    ## nivel 0.5 sobre-prediciendo y nivel 0.5 sub-prediciendo: 0.5 cada uno
    esperado = (0.5 * 1.0 + 0.5 * 1.0) / 2
    assert mean_pinball(observado, cuantiles, [0.5, 0.5]) == pytest.approx(esperado)


def test_crps_de_un_ensamble_degenerado_es_el_error_absoluto():
    ## todos los miembros iguales: la distribución es un punto y el CRPS es |x - y|
    miembros = np.full((5, 3), 2.0)
    observado = np.array([5.0, 5.0, 5.0])
    assert crps_ensemble(observado, miembros) == pytest.approx(3.0)


def test_crps_justo_corrige_el_sesgo_por_tamano():
    ## dos miembros en 0 y 2, observación en 1
    miembros = np.array([[0.0], [2.0]])
    observado = np.array([1.0])
    ## sesgado: 1 - (|0-2| + |2-0|) / (2*4) = 1 - 0.5
    assert crps_ensemble(observado, miembros, fair=False) == pytest.approx(0.5)
    ## justo: 1 - 4 / (2*2*1) = 0
    assert crps_ensemble(observado, miembros, fair=True) == pytest.approx(0.0)


def test_puntaje_de_intervalo_sin_violacion_es_el_ancho():
    observado = np.array([5.0])
    assert interval_score(observado, np.array([0.0]), np.array([10.0]),
                          0.1)[0] == pytest.approx(10.0)


def test_puntaje_de_intervalo_cobra_la_violacion():
    observado = np.array([12.0])
    ## ancho 10, más (2/0.1) * 2 de exceso = 10 + 40
    assert interval_score(observado, np.array([0.0]), np.array([10.0]),
                          0.1)[0] == pytest.approx(50.0)


def test_el_intervalo_del_cinco_al_noventaicinco_es_de_noventa():
    """La multa tiene que calcularse con alfa = 0.10, no con 0.05."""
    observado = np.array([12.0])
    con_noventa = interval_score(observado, np.array([0.0]), np.array([10.0]), 0.10)
    con_noventaicinco = interval_score(observado, np.array([0.0]),
                                       np.array([10.0]), 0.05)
    assert con_noventa[0] == pytest.approx(50.0)
    assert con_noventaicinco[0] == pytest.approx(90.0)


def test_wis_de_un_pronostico_perfecto_es_cero():
    niveles = [0.1, 0.25, 0.5, 0.75, 0.9]
    observado = np.array([4.0])
    cuantiles = np.full((len(niveles), 1), 4.0)
    assert weighted_interval_score(observado, cuantiles, niveles) == pytest.approx(0.0)


def test_wis_y_pinball_miden_lo_mismo_salvo_un_factor():
    """El puntaje de intervalo ponderado es, salvo escala, el pinball medio."""
    rng = np.random.default_rng(0)
    niveles = [0.1, 0.25, 0.5, 0.75, 0.9]
    observado = rng.normal(size=24) * 10
    base = np.sort(rng.normal(size=(len(niveles), 24)) * 5, axis=0)
    razon = (weighted_interval_score(observado, base, niveles)
             / mean_pinball(observado, base, niveles))
    assert 1.5 < razon < 2.5
