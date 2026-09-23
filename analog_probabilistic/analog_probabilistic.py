"""
Pronóstico probabilístico por análogos.

El método busca en la historia los tramos más parecidos al presente, ajusta un
mapa de nivel entre cada análogo y el presente, y lo aplica a lo que siguió a
ese análogo. Cada análogo aporta una trayectoria completa; el conjunto de
trayectorias es el ensamble, y sus cuantiles empíricos son el pronóstico.

El mapa de miembro admite tres pendientes, que son tres métodos distintos de la
literatura:

    'raw'      b = 1                     ensamble de análogos sin transformar
    'rescale'  b = sy / sx               reescalado de patrón
    'ols'      b = rho * sy / sx         mínimos cuadrados

Las tres comparten búsqueda, horizonte y costo, así que se pueden comparar sin
confundir el efecto de la selección con el del mapa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

AVAILABLE_MAPS = ("ols", "rescale", "raw")


@dataclass
class Analogs:
    """Análogos recuperados para un origen: dónde están y qué siguió a cada uno."""

    positions: np.ndarray
    windows: np.ndarray
    futures: np.ndarray
    similarities: np.ndarray


## ---------------------------------------------------------------------------
## Búsqueda de análogos
## ---------------------------------------------------------------------------

def find_analogs(
    series: np.ndarray,
    window: int,
    horizon: int,
    k: int = 40,
    separation: float = 0.5,
) -> Analogs:
    """
    Devuelve los k tramos pasados más correlacionados con el presente.

    La regla de separación descarta un candidato que empiece a menos de
    ``separation * window`` posiciones de uno ya aceptado, para que el ensamble
    no se llene de copias corridas del mismo episodio.
    """
    n = len(series)
    if n < 2 * window + 1:
        raise ValueError("serie demasiado corta para la ventana y el horizonte")

    present = series[n - window:]
    if np.std(present) == 0:
        raise ValueError("la ventana presente es constante")

    ## una sola pasada vectorizada sobre todas las ventanas candidatas: con
    ## series de decenas de miles de puntos, el lazo equivalente en Python
    ## domina el tiempo del método entero
    last = n - 2 * window
    strides = np.lib.stride_tricks.sliding_window_view(series[:last + window],
                                                       window)
    candidates = strides[:last]
    centered = candidates - candidates.mean(axis=1, keepdims=True)
    scales = np.sqrt((centered ** 2).sum(axis=1))
    present_centered = present - present.mean()
    present_scale = np.sqrt((present_centered ** 2).sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = centered @ present_centered / (scales * present_scale)
    similarity[~np.isfinite(similarity)] = -np.inf

    positions: list[int] = []
    gap = separation * window
    for pos in np.argsort(-similarity, kind="stable"):
        if similarity[pos] <= 0:
            break
        if any(abs(pos - taken) < gap for taken in positions):
            continue
        positions.append(int(pos))
        if len(positions) == k:
            break

    if not positions:
        raise ValueError("ningún análogo con correlación positiva")

    idx = np.array(positions)
    return Analogs(
        positions=idx,
        windows=np.array([series[p:p + window] for p in idx]),
        futures=np.array([series[p + window:p + window + horizon] for p in idx]),
        similarities=similarity[idx],
    )


## ---------------------------------------------------------------------------
## Mapa de miembro y ensamble
## ---------------------------------------------------------------------------

def member_map(
    analog_window: np.ndarray,
    present: np.ndarray,
    similarity: float,
    map_name: str = "ols",
) -> tuple[float, float]:
    """Devuelve la ordenada y la pendiente que llevan el análogo al presente."""
    if map_name not in AVAILABLE_MAPS:
        raise ValueError(f"mapa desconocido: {map_name}")

    if map_name == "raw":
        return 0.0, 1.0

    scale_x = float(np.std(analog_window))
    if scale_x == 0:
        return 0.0, 1.0

    slope = float(np.std(present)) / scale_x
    if map_name == "ols":
        slope *= similarity

    intercept = float(np.mean(present)) - slope * float(np.mean(analog_window))
    return intercept, slope


def build_ensemble(
    series: np.ndarray,
    window: int,
    horizon: int,
    k: int = 40,
    separation: float = 0.5,
    map_name: str = "ols",
    clip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye el ensamble de trayectorias y los pesos de sus miembros.

    Con ``clip`` activo, la entrada del mapa se acota al rango histórico y la
    salida a ``[percentil 1 - 2 sd, máximo]``. El tope superior es el máximo
    observado, de modo que el método no puede anunciar un precio récord: es una
    decisión conservadora que censura la cola derecha y conviene apagar cuando
    lo que se busca es justamente avisar de un extremo.
    """
    analogs = find_analogs(series, window, horizon, k, separation)
    present = series[len(series) - window:]

    low_input, high_input = float(series.min()), float(series.max())
    low_output = float(np.percentile(series, 1)) - 2.0 * float(np.std(series))
    high_output = float(series.max())

    members = []
    for analog_window, future, similarity in zip(
        analogs.windows, analogs.futures, analogs.similarities
    ):
        intercept, slope = member_map(analog_window, present, similarity, map_name)
        projected = future if not clip else np.clip(future, low_input, high_input)
        member = intercept + slope * projected
        if clip:
            member = np.clip(member, low_output, high_output)
        members.append(member)

    return np.array(members), analogs.similarities


## ---------------------------------------------------------------------------
## Distribución predictiva
## ---------------------------------------------------------------------------

def ensemble_quantiles(
    ensemble: np.ndarray,
    levels: Sequence[float],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Cuantiles empíricos del ensamble, paso a paso del horizonte.

    Con k miembros la distribución es una escalera de k peldaños, así que los
    niveles fuera de [1/(k+1), k/(k+1)] no se estiman: se interpolan.
    """
    levels = np.asarray(levels, dtype=float)
    if weights is None:
        return np.quantile(ensemble, levels, axis=0)

    weights = np.asarray(weights, dtype=float)
    quantiles = np.empty((len(levels), ensemble.shape[1]))
    for step in range(ensemble.shape[1]):
        order = np.argsort(ensemble[:, step])
        values, w = ensemble[order, step], weights[order]
        position = (np.cumsum(w) - 0.5 * w) / w.sum()
        quantiles[:, step] = np.interp(levels, position, values)
    return quantiles


def exceedance_probability(ensemble: np.ndarray, threshold: float) -> np.ndarray:
    """Fracción de miembros que superan el umbral en cada paso del horizonte."""
    return (ensemble > threshold).mean(axis=0)


def calibrate_quantiles(
    median: np.ndarray,
    spread: np.ndarray,
    standardized_errors: np.ndarray,
    levels: Sequence[float],
) -> np.ndarray:
    """
    Recalibra los cuantiles con los errores estandarizados de orígenes pasados.

    El ensamble de análogos resulta estrecho porque sus miembros se eligieron
    por parecerse al presente y por tanto se parecen entre sí. Reescalar la
    dispersión con errores ya observados devuelve cobertura a costa de ancho.
    """
    reference = np.quantile(np.asarray(standardized_errors), np.asarray(levels))
    return np.sort(median[None, :] + spread[None, :] * reference[:, None], axis=0)


## ---------------------------------------------------------------------------
## Interfaz de alto nivel
## ---------------------------------------------------------------------------

class AnalogProbabilistic:
    """
    Pronóstico probabilístico por análogos sobre una serie de una variable.

    Ejemplo:

        model = AnalogProbabilistic(window=48, k=40)
        model.fit(prices)
        quantiles = model.quantiles(horizon=24, levels=[0.05, 0.5, 0.95])
    """

    def __init__(
        self,
        window: int = 48,
        k: int = 40,
        separation: float = 0.5,
        map_name: str = "ols",
        clip: bool = True,
        weighted: bool = False,
    ):
        self.window = window
        self.k = k
        self.separation = separation
        self.map_name = map_name
        self.clip = clip
        self.weighted = weighted
        self.series: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (f"AnalogProbabilistic(window={self.window}, k={self.k}, "
                f"map_name='{self.map_name}')")

    def fit(self, series: np.ndarray) -> "AnalogProbabilistic":
        """Guarda la historia disponible hasta el origen del pronóstico."""
        self.series = np.asarray(series, dtype=float)
        return self

    def ensemble(self, horizon: int) -> np.ndarray:
        """Trayectorias del ensamble, una por análogo."""
        if self.series is None:
            raise ValueError("el modelo no tiene serie: falta llamar a fit")
        if horizon > self.window:
            raise ValueError("el horizonte no puede exceder la ventana presente")

        members, similarities = build_ensemble(
            self.series, self.window, horizon, self.k, self.separation,
            self.map_name, self.clip,
        )
        self._similarities = similarities
        return members

    def quantiles(self, horizon: int, levels: Sequence[float]) -> np.ndarray:
        """Cuantiles del ensamble para los niveles pedidos."""
        members = self.ensemble(horizon)
        weights = None
        if self.weighted:
            weights = self._similarities / self._similarities.sum()
        return ensemble_quantiles(members, levels, weights)

    def exceedance(self, horizon: int, threshold: float) -> np.ndarray:
        """Probabilidad de que la serie supere el umbral en cada paso."""
        return exceedance_probability(self.ensemble(horizon), threshold)
