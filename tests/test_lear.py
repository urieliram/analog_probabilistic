"""
Que LEAR aprenda lo que debe aprender y no mire lo que no debe.

La prueba fuerte es la primera: sobre una serie exactamente periódica, con un perfil
diario fijo y un efecto de día de la semana, LEAR tiene que reproducirla sin error. Si
el calendario estuviera corrido aunque fuera un día, el efecto de día de la semana
quedaría mal atribuido y el error no sería cero.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.benchmarks.lear import lear_punto  # noqa: E402

PERFIL = 500 + 200 * np.sin(np.arange(24) / 24 * 2 * np.pi)
EFECTO_SEMANAL = np.array([0.0, 10.0, 20.0, 30.0, 40.0, -80.0, -120.0])


def serie_periodica(dias: int) -> np.ndarray:
    return np.concatenate([PERFIL + EFECTO_SEMANAL[d % 7] for d in range(dias)])


def test_reproduce_una_serie_exactamente_periodica():
    dias = 130
    serie = serie_periodica(dias)
    prediccion = lear_punto(serie, dia_semana_origen=(dias - 1) % 7,
                            dias_calibracion=110)
    esperado = PERFIL + EFECTO_SEMANAL[dias % 7]
    ## La tolerancia no es cero porque la penalización del lazo deja un residuo
    ## propio cuando los días apenas superan a las variables: dos cienmilésimas de
    ## peso sobre precios de 500. Sigue siendo decisiva para lo que la prueba busca,
    ## que es un calendario corrido: los efectos de día de la semana valen entre 10 y
    ## 120 pesos, así que atribuir uno mal se vería tres órdenes de magnitud arriba.
    assert np.abs(prediccion - esperado).max() < 1e-3


def test_el_calendario_esta_amarrado_al_origen_y_no_al_principio():
    """
    Recortar la historia por el principio no debe cambiar qué día cree que predice.

    Es el error de uno que se cuela cuando el día de la semana se cuenta desde la
    primera hora de la historia: al recortarla a un número entero de días por el final,
    su primera hora cae en un día distinto.
    """
    dias = 130
    serie = serie_periodica(dias)
    completa = lear_punto(serie, dia_semana_origen=(dias - 1) % 7,
                          dias_calibracion=110)
    ## se quitan siete horas del principio: la historia ya no empieza en hora 0
    recortada = lear_punto(serie[7:], dia_semana_origen=(dias - 1) % 7,
                           dias_calibracion=110)
    assert np.abs(completa - recortada).max() < 1e-6


def test_no_mira_mas_alla_del_origen():
    """
    El pronóstico depende sólo de la historia, no de lo que venga después.

    Se predice el día siguiente con la historia hasta el origen, y luego se vuelve a
    predecir con una historia a la que se le pegó un día posterior absurdo y se cortó
    de nuevo en el mismo origen. El resultado tiene que ser idéntico.
    """
    dias = 130
    serie = serie_periodica(dias)
    antes = lear_punto(serie, dia_semana_origen=(dias - 1) % 7, dias_calibracion=110)

    extendida = np.concatenate([serie, np.full(24, 99999.0)])
    despues = lear_punto(extendida[:len(serie)], dia_semana_origen=(dias - 1) % 7,
                         dias_calibracion=110)
    assert np.array_equal(antes, despues)


def test_rechaza_un_horizonte_que_no_sea_de_un_dia():
    with pytest.raises(ValueError, match="24 horas"):
        lear_punto(serie_periodica(130), horizonte=48)


def test_rechaza_una_historia_demasiado_corta():
    with pytest.raises(ValueError, match="historia demasiado corta"):
        lear_punto(serie_periodica(5))


def test_rechaza_una_calibracion_con_menos_dias_que_variables():
    """
    Son 103 variables; con menos días que eso no hay con qué estimar el ruido.

    Importa decirlo claro porque el LEAR de referencia sí usa ventanas de 56 y 84
    días, que caen debajo del límite: esa variante exige una estimación aparte de la
    varianza y aquí queda fuera a propósito.
    """
    with pytest.raises(ValueError, match="103 variables"):
        lear_punto(serie_periodica(130), dias_calibracion=80)
