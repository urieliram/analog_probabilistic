"""
Produce pronósticos y los guarda. No calcula ningún puntaje.

Es la primera mitad del experimento: recorre el panel y los orígenes del tramo
pedido, pide a cada método sus cuantiles, y los escribe en el almacén. Calificar es
otro paso, `evaluate.py`, y por eso cambiar una métrica ya no obliga a volver a
correr nada.

La configuración del método NO se escribe aquí: se lee de la que quedó congelada en
el tramo de selección. Sin ese archivo, esto no arranca.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_probabilistic.analog_probabilistic import (  # noqa: E402
    build_ensemble, ensemble_quantiles)
from experiments.forecast_store import ForecastStore  # noqa: E402
from experiments.protocol import (HORIZON, RESULTS, SEARCH_YEARS,  # noqa: E402
                                  SELECTION_END, SELECTION_START, TEST_END,
                                  TEST_START, build_origins, load_selected,
                                  load_selected_zones)
from experiments.scores import LEVELS  # noqa: E402
from experiments.series import load_series  # noqa: E402

TRAMOS = {
    "seleccion": (SELECTION_START, SELECTION_END),
    "prueba": (TEST_START, TEST_END),
}


def pronostico_analogo(historia: np.ndarray, config: dict, gamma: float = None):
    """Cuantiles y miembros del ensamble de análogos."""
    miembros, _ = build_ensemble(
        historia, window=int(config["window"]), horizon=HORIZON,
        k=int(config["k"]), separation=float(config["separation"]),
        gamma=float(config["gamma"] if gamma is None else gamma))
    return ensemble_quantiles(miembros, LEVELS), miembros


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tramo", choices=sorted(TRAMOS), default="prueba")
    parser.add_argument("--salida", default=None)
    parser.add_argument("--zonas", type=int, default=None,
                        help="limita el panel, para pruebas rápidas")
    args = parser.parse_args()

    config = load_selected()
    zonas = load_selected_zones()["zonas"][: args.zonas]
    inicio_tramo, fin_tramo = TRAMOS[args.tramo]

    print(f"tramo {args.tramo}: {inicio_tramo} a {fin_tramo}")
    print(f"configuración congelada: "
          f"{ {k: config[k] for k in ('window', 'k', 'separation', 'gamma')} }")
    print(f"{len(zonas)} zonas", flush=True)

    destino = Path(args.salida or RESULTS / f"forecasts_{args.tramo}.parquet")
    almacen = ForecastStore(LEVELS, destino)

    ## la variante sin contracción se guarda al lado: es la comparación que
    ## sostiene la tesis y sale gratis, porque comparte la búsqueda de análogos
    variantes = {"Analog": None, "Analog-sin-contraccion": 0.0}

    comienzo = time.time()
    for numero, zona in enumerate(zonas, 1):
        serie = load_series("pml", zona)
        origenes = build_origins(inicio_tramo, fin_tramo, serie)
        for origen in origenes:
            historia = serie.loc[:origen].values[-SEARCH_YEARS * 8760:]
            observado = serie.loc[origen + pd.Timedelta(hours=1):
                                  origen + pd.Timedelta(hours=HORIZON)].values
            if len(observado) != HORIZON:
                continue
            for nombre, gamma in variantes.items():
                reloj = time.time()
                try:
                    cuantiles, miembros = pronostico_analogo(historia, config,
                                                             gamma)
                except ValueError:
                    continue
                almacen.add(zona, origen, nombre, cuantiles, observado,
                            miembros, time.time() - reloj)
        print(f"  {numero}/{len(zonas)} {zona:20s} "
              f"{time.time() - comienzo:6.0f}s", flush=True)

    escrito = almacen.save()
    print(f"\n{escrito}")


if __name__ == "__main__":
    main()
