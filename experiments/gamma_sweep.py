"""
Dónde conviene poner el exponente de contracción.

La hipótesis del artículo es que el pronóstico probabilístico quiere menos
contracción que el puntual: los análogos se seleccionaron por tener correlación
alta, y volver a usar esa correlación para encoger la pendiente estrecha un
ensamble que ya venía estrecho. Es falsable: basta que el gamma elegido salga
cerca de uno.

El barrido corre sobre el tramo de selección y el panel congelado. La búsqueda de
análogos se hace UNA vez por origen y zona, y los once valores de gamma se aplican
al mismo resultado: la búsqueda es todo el costo del método.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_probabilistic.analog_probabilistic import (  # noqa: E402
    ensemble_quantiles, find_analogs, member_map)
from experiments.protocol import (GAMMA_GRID, HORIZON, RESULTS,  # noqa: E402
                                  SEARCH_YEARS, SELECTION_END, SELECTION_START,
                                  build_origins, load_selected_zones)
from experiments.scores import LEVELS, evaluate  # noqa: E402
from experiments.series import load_series  # noqa: E402

WINDOW = 48
K = 40
SEPARATION = 0.5


def miembros_por_gamma(historia, analogos, presente, gammas, limites):
    """Aplica cada gamma al mismo conjunto de análogos ya encontrado."""
    entrada_baja, entrada_alta, salida_baja, salida_alta = limites
    por_gamma = {}
    for gamma in gammas:
        miembros = []
        for ventana, futuro, parecido in zip(analogos.windows, analogos.futures,
                                             analogos.similarities):
            ordenada, pendiente = member_map(ventana, presente, parecido,
                                             gamma=gamma)
            proyectado = np.clip(futuro, entrada_baja, entrada_alta)
            miembros.append(np.clip(ordenada + pendiente * proyectado,
                                    salida_baja, salida_alta))
        por_gamma[gamma] = np.array(miembros)
    return por_gamma


def main() -> None:
    zonas = load_selected_zones()["zonas"]
    print(f"{len(zonas)} zonas del panel · gamma de "
          f"{GAMMA_GRID[0]} a {GAMMA_GRID[-1]}", flush=True)

    filas = []
    inicio = time.time()
    for numero, zona in enumerate(zonas, 1):
        serie = load_series("pml", zona)
        origenes = build_origins(SELECTION_START, SELECTION_END, serie)

        for origen in origenes:
            historia = serie.loc[:origen].values[-SEARCH_YEARS * 8760:]
            observado = serie.loc[origen + pd.Timedelta(hours=1):
                                  origen + pd.Timedelta(hours=HORIZON)].values
            if len(observado) != HORIZON or len(historia) < 2 * WINDOW + HORIZON:
                continue

            try:
                analogos = find_analogs(historia, WINDOW, HORIZON, K, SEPARATION)
            except ValueError:
                continue
            presente = historia[len(historia) - WINDOW:]
            limites = (float(historia.min()), float(historia.max()),
                       float(np.percentile(historia, 1)) - 2 * float(historia.std()),
                       float(historia.max()))

            for gamma, miembros in miembros_por_gamma(
                    historia, analogos, presente, GAMMA_GRID, limites).items():
                cuantiles = ensemble_quantiles(miembros, LEVELS)
                filas.append({"zona": zona, "origen": origen, "gamma": gamma,
                              "k_real": len(miembros),
                              **evaluate(observado, cuantiles, LEVELS, miembros)})

        print(f"  {numero}/{len(zonas)} {zona:20s} "
              f"{time.time() - inicio:6.0f}s", flush=True)

    tabla = pd.DataFrame(filas)
    RESULTS.mkdir(exist_ok=True)
    tabla.to_csv(RESULTS / "gamma_sweep_raw.csv", index=False)

    resumen = tabla.groupby("gamma")[
        ["mean_pinball", "crps", "reliability", "sharp90", "cover90",
         "winkler90", "mae", "pinball95", "k_real"]].mean()
    resumen.to_csv(RESULTS / "gamma_sweep.csv")
    print("\n", resumen.round(4).to_string(), sep="")
    print(f"\norígenes por zona: {tabla.origen.nunique()} · "
          f"filas: {len(tabla):,d}")


if __name__ == "__main__":
    main()
