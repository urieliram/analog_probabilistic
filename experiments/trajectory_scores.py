"""
Puntajes que miran el camino, no cada hora por separado.

Es lo único que el método tiene y los comparativos no: cada miembro es una
trayectoria completa de veinticuatro horas, así que el ensamble carga la
dependencia entre horas. Afirmarlo sin medirlo no vale, y medirlo exige dos
cuidados:

  tamaño del ensamble   el puntaje de energía empírico depende de cuántos miembros
                        haya, así que un método con cuarenta y otro con cien no se
                        comparan en crudo. Se usa la forma corregida.

  los que no tienen caminos  a un método que sólo entrega cuantiles marginales hay
                        que construírselos, y cómo se haga decide el resultado. Se
                        hace de dos maneras, muestreo independiente y cópula
                        gaussiana con la correlación de sus errores anteriores al
                        origen, y se reportan las dos.
"""

from typing import Optional

import numpy as np


def energy_score(observed: np.ndarray, members: np.ndarray,
                 fair: bool = True) -> float:
    """
    Puntaje de energía de un ensamble de trayectorias.

    Es el CRPS llevado a varias dimensiones: premia acercarse a lo observado y
    castiga que los miembros se amontonen. Con ``fair`` el segundo término lleva
    denominador m(m-1), que corrige el sesgo por tamaño del ensamble.
    """
    members = np.asarray(members, dtype=float)
    m = members.shape[0]
    distancia_al_dato = np.linalg.norm(members - observed[None, :], axis=1).mean()
    if m < 2:
        return float(distancia_al_dato)

    diferencias = members[:, None, :] - members[None, :, :]
    dispersion = np.linalg.norm(diferencias, axis=2).sum()
    denominador = 2 * m * (m - 1) if fair else 2 * m * m
    return float(distancia_al_dato - dispersion / denominador)


def variogram_score(observed: np.ndarray, members: np.ndarray,
                    order: float = 0.5,
                    weights: Optional[np.ndarray] = None) -> float:
    """
    Puntaje de variograma: compara la estructura de dependencia, no el nivel.

    Mira, para cada par de horas, cuánto se separan en lo observado y cuánto se
    separan en promedio en el ensamble. Es menos sensible al sesgo y más sensible a
    la dependencia que el puntaje de energía, que es justo lo que aquí se afirma.
    """
    members = np.asarray(members, dtype=float)
    h = members.shape[1]
    observadas = np.abs(observed[:, None] - observed[None, :]) ** order
    previstas = np.abs(members[:, :, None] - members[:, None, :]) ** order
    diferencia = (observadas - previstas.mean(axis=0)) ** 2

    if weights is None:
        ## las horas vecinas pesan más: la dependencia que importa es la cercana
        distancia = np.abs(np.arange(h)[:, None] - np.arange(h)[None, :])
        weights = np.divide(1.0, distancia, out=np.zeros_like(distancia,
                                                              dtype=float),
                            where=distancia > 0)
    return float((weights * diferencia).sum())


def paths_independent(quantiles: np.ndarray, levels, n_paths: int = 100,
                      seed: int = 0) -> np.ndarray:
    """
    Caminos para un método que sólo da cuantiles marginales, hora por hora.

    Cada hora se sortea por su cuenta: es el escenario sin ninguna estructura de
    dependencia, el piso contra el que medirse.
    """
    rng = np.random.default_rng(seed)
    niveles = np.asarray(levels, dtype=float)
    pasos = quantiles.shape[1]
    caminos = np.empty((n_paths, pasos))
    for paso in range(pasos):
        u = rng.uniform(niveles[0], niveles[-1], n_paths)
        caminos[:, paso] = np.interp(u, niveles, quantiles[:, paso])
    return caminos


def paths_gaussian_copula(quantiles: np.ndarray, levels, correlation: np.ndarray,
                          n_paths: int = 100, seed: int = 0) -> np.ndarray:
    """
    Caminos con dependencia, lo mejor que ese método puede hacer sin cambiar.

    La correlación tiene que venir de errores ANTERIORES al origen; estimarla una
    sola vez sobre todo el periodo de prueba sería mirar hacia adelante.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    niveles = np.asarray(levels, dtype=float)
    pasos = quantiles.shape[1]

    ## una matriz de correlación estimada con pocos datos puede no ser definida
    ## positiva: se recorta el espectro antes de factorizar
    valores, vectores = np.linalg.eigh(correlation)
    saneada = vectores @ np.diag(np.clip(valores, 1e-8, None)) @ vectores.T
    escala = np.sqrt(np.diag(saneada))
    saneada = saneada / np.outer(escala, escala)

    normales = rng.multivariate_normal(np.zeros(pasos), saneada, size=n_paths)
    uniformes = np.clip(norm.cdf(normales), niveles[0], niveles[-1])
    caminos = np.empty((n_paths, pasos))
    for paso in range(pasos):
        caminos[:, paso] = np.interp(uniformes[:, paso], niveles,
                                     quantiles[:, paso])
    return caminos
