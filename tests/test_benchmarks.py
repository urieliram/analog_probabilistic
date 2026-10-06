"""Pruebas de los comparativos simples."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.benchmarks.naive import (naive, naive_combinado,  # noqa: E402
                                          naive_semanal)

NIVELES = [0.1, 0.25, 0.5, 0.75, 0.9]


def serie_perfectamente_diaria(n=2000):
    return np.tile(np.arange(24, dtype=float), n // 24 + 1)[:n]


def test_sobre_una_serie_que_se_repite_el_naive_acierta_y_no_duda():
    """Si el día de ayer es idéntico al de hoy, el intervalo tiene que ser nulo."""
    serie = serie_perfectamente_diaria()
    cuantiles = naive(serie, 24, NIVELES)
    assert np.allclose(cuantiles[0], cuantiles[-1])
    assert np.allclose(cuantiles[2], serie[-24:])


def test_el_semanal_parte_del_mismo_dia_de_la_semana_pasada():
    """Sin errores que corregir, el semanal repite las horas de hace una semana."""
    ## serie con periodo de una semana: el semanal acierta y el diario no
    patron = np.arange(168, dtype=float)
    serie = np.tile(patron, 20)
    centro = naive_semanal(serie, 24, [0.5])[0]
    assert np.allclose(centro, serie[-168:-144])


def test_el_diario_falla_donde_el_semanal_acierta():
    """La razón de que el semanal esté en el cuadro: la semana manda."""
    serie = np.tile(np.arange(168, dtype=float), 20)
    objetivo = serie[-168:-144]
    error_semanal = np.abs(naive_semanal(serie, 24, [0.5])[0] - objetivo).mean()
    error_diario = np.abs(naive(serie, 24, [0.5])[0] - objetivo).mean()
    assert error_semanal < error_diario


def test_los_cuantiles_salen_ordenados():
    rng = np.random.default_rng(1)
    serie = np.cumsum(rng.normal(size=3000)) + 500
    for metodo in (naive, naive_semanal, naive_combinado):
        cuantiles = metodo(serie, 24, NIVELES)
        assert np.all(np.diff(cuantiles, axis=0) >= -1e-9), metodo.__name__


def test_el_combinado_promedia_a_sus_dos_padres():
    """
    Sobre una serie sin error, los tres coinciden y el combinado es el promedio.

    Con una serie cualquiera no vale comparar medianas, porque cada método suma
    los cuantiles de SUS propios errores y ésos no se promedian: la prueba sería
    sobre otra cosa.
    """
    serie = serie_perfectamente_diaria()
    diario = naive(serie, 24, [0.5])[0]
    semanal = naive_semanal(serie, 24, [0.5])[0]
    combinado = naive_combinado(serie, 24, [0.5])[0]
    assert np.allclose(combinado, (diario + semanal) / 2)
    assert np.allclose(combinado, serie[-24:])
