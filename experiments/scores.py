"""Puntajes de un pronóstico probabilístico y la prueba de Diebold-Mariano."""

from typing import Dict, Sequence

import numpy as np
from scipy import stats

LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def pinball(observed: np.ndarray, quantile: np.ndarray, level: float) -> np.ndarray:
    """Pérdida pinball de un cuantil, hora por hora."""
    error = observed - quantile
    return np.where(error >= 0, level * error, (level - 1) * error)


def evaluate(observed: np.ndarray, quantiles: np.ndarray,
             levels: Sequence[float] = LEVELS) -> Dict[str, float]:
    """
    Puntajes de un pronóstico de un origen.

    ``quantiles`` trae una fila por nivel, en el mismo orden que ``levels``, y
    una columna por paso del horizonte.
    """
    levels = np.asarray(levels, dtype=float)
    position = {round(float(level), 2): row for row, level in enumerate(levels)}

    crps = np.mean([pinball(observed, quantiles[row], level)
                    for row, level in enumerate(levels)])
    below = {level: float(np.mean(observed <= quantiles[row]))
             for row, level in enumerate(levels)}
    reliability = float(np.mean([abs(level - below[level]) for level in levels]))

    median = quantiles[position[0.50]]
    low80, high80 = quantiles[position[0.10]], quantiles[position[0.90]]
    low95, high95 = quantiles[position[0.05]], quantiles[position[0.95]]

    ## el puntaje de Winkler cobra el ancho del intervalo más una multa por
    ## cada observación que se sale: es el que castiga prometer certeza
    winkler = np.mean(
        (high95 - low95)
        + (2 / 0.10) * (low95 - observed) * (observed < low95)
        + (2 / 0.10) * (observed - high95) * (observed > high95)
    )

    scores = {
        "crps": float(crps),
        "reliability": reliability,
        "sharp80": float(np.mean(high80 - low80)),
        "sharp95": float(np.mean(high95 - low95)),
        "cover80": float(np.mean((observed >= low80) & (observed <= high80))),
        "cover95": float(np.mean((observed >= low95) & (observed <= high95))),
        "winkler95": float(winkler),
        "mae": float(np.mean(np.abs(observed - median))),
        "rmse": float(np.sqrt(np.mean((observed - median) ** 2))),
        "pinball95": float(np.mean(pinball(observed, quantiles[position[0.95]], 0.95))),
    }
    scores.update({f"hit_{level:g}": below[level] for level in levels})
    return scores


def diebold_mariano(differences: np.ndarray, lags: int = 3) -> tuple:
    """
    Prueba de una cola sobre la diferencia de pérdidas por origen.

    La varianza de largo plazo es de Newey-West, porque orígenes consecutivos
    comparten días y sus pérdidas están correlacionadas. Un estadístico
    negativo favorece al primer método de la diferencia.
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
