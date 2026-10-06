"""
Los comparativos que todo pronóstico de precio tiene que ganar.

No son adornos: en precios eléctricos el naive de 168 horas —el mismo día de la
semana pasada— es sorprendentemente difícil de batir, porque la semana manda sobre
el precio tanto como el día.

Los tres dan cuantiles a partir de sus propios errores recientes, medidos paso por
paso del horizonte y siempre con datos anteriores al origen.
"""

from typing import Sequence

import numpy as np

VENTANA_ERRORES = 60


def _cuantiles_de_errores(centro: np.ndarray, errores: np.ndarray,
                          levels: Sequence[float]) -> np.ndarray:
    """Suma al pronóstico central los cuantiles de sus errores por paso."""
    if errores.size == 0:
        return np.tile(centro, (len(levels), 1))
    desplazamientos = np.quantile(errores, levels, axis=0)
    return np.sort(centro[None, :] + desplazamientos, axis=0)


def _errores_recientes(historia: np.ndarray, rezago: int, horizonte: int,
                       ventana: int) -> np.ndarray:
    """Errores del mismo rezago en los últimos días ya observados."""
    errores = []
    for atras in range(1, ventana + 1):
        fin = len(historia) - (atras - 1) * horizonte
        objetivo = historia[fin - horizonte:fin]
        referencia = historia[fin - horizonte - rezago:fin - rezago]
        if len(objetivo) == horizonte and len(referencia) == horizonte:
            errores.append(objetivo - referencia)
    return np.array(errores)


def naive(historia: np.ndarray, horizonte: int, levels: Sequence[float],
          rezago: int = 24, ventana: int = VENTANA_ERRORES) -> np.ndarray:
    """Repite lo que pasó hace ``rezago`` horas y pone intervalos con sus errores."""
    centro = historia[-rezago:][:horizonte].astype(float)
    if len(centro) < horizonte:
        centro = np.resize(centro, horizonte)
    return _cuantiles_de_errores(
        centro, _errores_recientes(historia, rezago, horizonte, ventana), levels)


def naive_semanal(historia: np.ndarray, horizonte: int,
                  levels: Sequence[float], **kwargs) -> np.ndarray:
    """El mismo día de la semana pasada."""
    return naive(historia, horizonte, levels, rezago=168, **kwargs)


def naive_combinado(historia: np.ndarray, horizonte: int,
                    levels: Sequence[float],
                    ventana: int = VENTANA_ERRORES) -> np.ndarray:
    """
    Promedio del día anterior y del mismo día de la semana pasada.

    El promedio de dos comparativos simples suele ganarle a los dos, y por eso
    entra: sin él, el artículo estaría eligiendo un rival débil.
    """
    diario = historia[-24:][:horizonte].astype(float)
    semanal = historia[-168:][:horizonte].astype(float)
    if len(diario) < horizonte:
        diario = np.resize(diario, horizonte)
    if len(semanal) < horizonte:
        semanal = np.resize(semanal, horizonte)
    centro = (diario + semanal) / 2

    e_diario = _errores_recientes(historia, 24, horizonte, ventana)
    e_semanal = _errores_recientes(historia, 168, horizonte, ventana)
    n = min(len(e_diario), len(e_semanal))
    errores = ((e_diario[:n] + e_semanal[:n]) / 2 if n else np.empty((0, horizonte)))
    return _cuantiles_de_errores(centro, errores, levels)
