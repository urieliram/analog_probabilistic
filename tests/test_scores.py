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


## ---------------------------------------------------------------------------
## Puntajes de trayectoria
## ---------------------------------------------------------------------------

from experiments.trajectory_scores import (energy_score,  # noqa: E402
                                           paths_gaussian_copula,
                                           paths_independent, variogram_score)


def test_el_puntaje_de_energia_de_un_ensamble_exacto_es_cero():
    observado = np.array([1.0, 2.0, 3.0])
    miembros = np.tile(observado, (8, 1))
    assert energy_score(observado, miembros) == pytest.approx(0.0)


def test_el_puntaje_de_energia_se_reduce_al_crps_en_una_dimension():
    from experiments.scores import crps_ensemble
    rng = np.random.default_rng(1)
    miembros = rng.normal(size=(30, 1))
    observado = np.array([0.3])
    assert energy_score(observado, miembros) == pytest.approx(
        crps_ensemble(observado, miembros), rel=1e-9)


def test_el_variograma_premia_tener_la_dependencia_correcta():
    """
    Dos ensambles con la misma distribución por hora y distinta dependencia.

    El que conserva la relación entre horas tiene que puntuar mejor; un puntaje
    que no distinga eso no sirve para sostener la afirmación del artículo.
    """
    rng = np.random.default_rng(5)
    base = rng.normal(size=100)
    ## lo observado sube suave: horas vecinas parecidas
    observado = np.cumsum(rng.normal(size=24)) * 0.1

    con_dependencia = np.array([np.cumsum(rng.normal(size=24)) * 0.1
                                for _ in range(100)])
    sin_dependencia = con_dependencia.copy()
    for paso in range(24):
        rng.shuffle(sin_dependencia[:, paso])

    assert (variogram_score(observado, con_dependencia)
            < variogram_score(observado, sin_dependencia))


def test_los_caminos_independientes_respetan_los_cuantiles_marginales():
    niveles = [0.1, 0.3, 0.5, 0.7, 0.9]
    cuantiles = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    caminos = paths_independent(cuantiles, niveles, n_paths=20000, seed=1)
    assert np.median(caminos) == pytest.approx(0.0, abs=0.05)
    assert caminos.min() >= -2.0 and caminos.max() <= 2.0


def test_la_copula_produce_caminos_correlacionados():
    niveles = [0.1, 0.5, 0.9]
    cuantiles = np.tile(np.array([[-1.0], [0.0], [1.0]]), (1, 3))
    correlacion = np.full((3, 3), 0.9)
    np.fill_diagonal(correlacion, 1.0)
    caminos = paths_gaussian_copula(cuantiles, niveles, correlacion,
                                    n_paths=4000, seed=2)
    observada = np.corrcoef(caminos.T)[0, 1]
    assert observada > 0.6


def test_el_crps_rapido_da_el_mismo_numero_que_comparar_todos_los_pares():
    """
    La forma ordenada del CRPS es exacta, no una aproximación.

    Calcular la dispersión comparando todos los pares cuesta m al cuadrado: con mil
    miembros la evaluación se vuelve impracticable y de hecho murió. La forma ordenada
    da el MISMO número en m log m, y esta prueba es lo que lo sostiene.
    """
    rng = np.random.default_rng(11)
    for m in (2, 3, 7, 40, 200):
        observado = rng.normal(500, 60, 12)
        miembros = observado[None, :] + rng.normal(0, 45, (m, 12))

        ## la forma por pares, escrita aquí a propósito para no depender del módulo
        termino = np.abs(miembros - observado[None, :]).mean(axis=0)
        pares = np.abs(miembros[:, None, :] - miembros[None, :, :]).sum(axis=(0, 1))
        for justo, den in [(True, 2 * m * (m - 1)), (False, 2 * m * m)]:
            esperado = float(np.mean(termino - pares / den))
            obtenido = crps_ensemble(observado, miembros, fair=justo)
            assert abs(obtenido - esperado) < 1e-9 * max(1.0, abs(esperado)), \
                f"m={m} justo={justo}: {obtenido} contra {esperado}"
