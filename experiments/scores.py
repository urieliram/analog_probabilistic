"""
Puntajes de un pronóstico probabilístico, y las pruebas que los comparan.

Tres advertencias que vienen de haber tenido mal estas definiciones:

1. El intervalo que va del cuantil 0.05 al 0.95 es el **central de 90%**, no de
   95%. Antes se llamaba de 95% y la multa de Winkler ya usaba alfa = 0.10, de
   modo que el número estaba bien y el nombre mal.
2. El promedio de la pérdida pinball sobre una rejilla de niveles **no es el
   CRPS**: le falta el factor dos y no cubre todo el intervalo unitario. Aquí se
   llama por su nombre, pérdida pinball media, y el CRPS se calcula aparte.
3. Comparar un CRPS exacto contra un promedio de pinball sobre nueve niveles es
   comparar escalas distintas. El puntaje primario del artículo se calcula para
   todos sobre la **misma rejilla**.
"""

from typing import Dict, Sequence

import numpy as np
from scipy import stats

LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def pinball(observed: np.ndarray, quantile: np.ndarray, level: float) -> np.ndarray:
    """Pérdida pinball de un cuantil, punto por punto."""
    error = observed - quantile
    return np.where(error >= 0, level * error, (level - 1) * error)


def mean_pinball(observed: np.ndarray, quantiles: np.ndarray,
                 levels: Sequence[float]) -> float:
    """
    Pérdida pinball promediada sobre los niveles dados y sobre el horizonte.

    Es el puntaje primario del artículo. Para que dos métodos sean comparables
    tienen que evaluarse sobre la misma rejilla de niveles: un modelo que entrega
    noventa y nueve cuantiles y otro que entrega nueve no se comparan calculando a
    cada uno con su propia resolución.
    """
    levels = np.asarray(levels, dtype=float)
    return float(np.mean([pinball(observed, quantiles[row], level)
                          for row, level in enumerate(levels)]))


def crps_ensemble(observed: np.ndarray, members: np.ndarray,
                  fair: bool = True) -> float:
    """
    CRPS empírico de un ensamble, en su forma por diferencias absolutas.

    Con ``fair`` se usa el denominador m(m-1), que corrige el sesgo por tamaño del
    ensamble. Importa aquí: un método con cuarenta miembros y otro con cien no son
    comparables con la forma sesgada.

    ``members`` tiene una fila por miembro y una columna por paso del horizonte.
    """
    members = np.asarray(members, dtype=float)
    m = members.shape[0]
    if m < 2:
        return float(np.mean(np.abs(members[0] - observed)))

    termino_error = np.abs(members - observed[None, :]).mean(axis=0)

    ## La dispersión es la suma de |xi - xj| sobre todos los pares, y calcularla por
    ## pares cuesta m al cuadrado: con mil miembros son un millón de comparaciones por
    ## paso del horizonte, y la evaluación de Moirai murió por eso. Ordenando, la misma
    ## suma exacta sale en una pasada:
    ##
    ##     suma de |xi - xj| = 2 * suma_i (2i - m - 1) * x_(i)
    ##
    ## con i de 1 a m sobre los valores ya ordenados. No es una aproximación ni un
    ## muestreo: da el mismo número, y el costo pasa de m al cuadrado a m log m.
    ordenados = np.sort(members, axis=0)
    pesos = (2 * np.arange(1, m + 1) - m - 1).astype(float)[:, None]
    dispersion = 2.0 * (ordenados * pesos).sum(axis=0)

    denominador = 2 * m * (m - 1) if fair else 2 * m * m
    return float(np.mean(termino_error - dispersion / denominador))


def interval_score(observed: np.ndarray, low: np.ndarray, high: np.ndarray,
                   alpha: float) -> np.ndarray:
    """Puntaje de intervalo de Winkler para un intervalo central de 1 - alfa."""
    return ((high - low)
            + (2 / alpha) * (low - observed) * (observed < low)
            + (2 / alpha) * (observed - high) * (observed > high))


def weighted_interval_score(observed: np.ndarray, quantiles: np.ndarray,
                            levels: Sequence[float]) -> float:
    """
    Puntaje de intervalo ponderado: aproxima al CRPS con cuantiles simétricos.

    Es lo que se usa para los métodos que sólo entregan cuantiles y no una muestra
    de la distribución.
    """
    levels = np.asarray(levels, dtype=float)
    posicion = {round(float(a), 3): fila for fila, a in enumerate(levels)}
    if 0.5 not in posicion:
        raise ValueError("el puntaje de intervalo ponderado necesita la mediana")

    alphas = sorted({round(2 * float(a), 3) for a in levels if a < 0.5})
    total = np.abs(observed - quantiles[posicion[0.5]]) / 2
    usados = 0
    for alpha in alphas:
        bajo, alto = round(alpha / 2, 3), round(1 - alpha / 2, 3)
        if bajo not in posicion or alto not in posicion:
            continue
        total = total + (alpha / 2) * interval_score(
            observed, quantiles[posicion[bajo]], quantiles[posicion[alto]], alpha)
        usados += 1
    return float(np.mean(total / (usados + 0.5)))


def evaluate(observed: np.ndarray, quantiles: np.ndarray,
             levels: Sequence[float] = LEVELS,
             members: np.ndarray = None) -> Dict[str, float]:
    """
    Puntajes de un pronóstico de un origen.

    ``quantiles`` trae una fila por nivel, en el orden de ``levels``, y una columna
    por paso del horizonte. ``members`` es opcional: cuando el método entrega una
    muestra, se añade el CRPS del ensamble.
    """
    levels = np.asarray(levels, dtype=float)
    posicion = {round(float(a), 3): fila for fila, a in enumerate(levels)}

    debajo = {float(a): float(np.mean(observed <= quantiles[fila]))
              for fila, a in enumerate(levels)}
    reliability = float(np.mean([abs(a - debajo[float(a)]) for a in levels]))

    puntajes = {
        "mean_pinball": mean_pinball(observed, quantiles, levels),
        "reliability": reliability,
        "mae": float(np.mean(np.abs(observed - quantiles[posicion[0.5]]))),
        "rmse": float(np.sqrt(np.mean(
            (observed - quantiles[posicion[0.5]]) ** 2))),
    }

    ## los intervalos se nombran por la cobertura que de verdad anuncian
    for bajo, alto, nombre in [(0.1, 0.9, "80"), (0.05, 0.95, "90"),
                               (0.025, 0.975, "95")]:
        if bajo not in posicion or alto not in posicion:
            continue
        inferior, superior = quantiles[posicion[bajo]], quantiles[posicion[alto]]
        alpha = round(2 * bajo, 3)
        puntajes[f"sharp{nombre}"] = float(np.mean(superior - inferior))
        puntajes[f"cover{nombre}"] = float(np.mean(
            (observed >= inferior) & (observed <= superior)))
        puntajes[f"winkler{nombre}"] = float(np.mean(
            interval_score(observed, inferior, superior, alpha)))

    if 0.95 in posicion:
        puntajes["pinball95"] = float(np.mean(
            pinball(observed, quantiles[posicion[0.95]], 0.95)))

    try:
        puntajes["wis"] = weighted_interval_score(observed, quantiles, levels)
    except ValueError:
        pass

    if members is not None:
        puntajes["crps"] = crps_ensemble(observed, members)

    puntajes.update({f"hit_{a:g}": debajo[float(a)] for a in levels})
    return puntajes


def diebold_mariano(differences: np.ndarray, lags: int = 3) -> tuple:
    """
    Prueba de una cola sobre la diferencia de pérdidas por origen.

    La varianza de largo plazo es de Newey-West, porque orígenes consecutivos
    comparten días y sus pérdidas están correlacionadas. Un estadístico negativo
    favorece al primer método de la diferencia.
    """
    n = len(differences)
    mean = differences.mean()
    variance = np.sum((differences - mean) ** 2) / n
    for lag in range(1, lags + 1):
        covariance = np.sum(
            (differences[lag:] - mean) * (differences[:-lag] - mean)) / n
        variance += 2 * (1 - lag / (lags + 1)) * covariance
    if variance <= 0:
        return float("nan"), float("nan")
    statistic = mean / np.sqrt(variance / n)
    return float(statistic), float(stats.norm.cdf(statistic))
