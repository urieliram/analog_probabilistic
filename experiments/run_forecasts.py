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
    ensemble_quantiles, find_analogs, member_map)
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


def ensambles_por_gamma(historia: np.ndarray, config: dict, gammas: dict):
    """
    Un ensamble por cada valor de gamma, con UNA sola búsqueda de análogos.

    La búsqueda es todo el costo del método: repetirla por variante duplicaba el
    tiempo de la corrida sin aportar nada, porque las variantes comparten los
    mismos análogos y sólo difieren en la pendiente del mapa.
    """
    ventana = int(config["window"])
    analogos = find_analogs(historia, ventana, HORIZON, int(config["k"]),
                            float(config["separation"]))
    presente = historia[len(historia) - ventana:]

    entrada = (float(historia.min()), float(historia.max()))
    salida = (float(np.percentile(historia, 1)) - 2 * float(historia.std()),
              float(historia.max()))

    salidas = {}
    for nombre, gamma in gammas.items():
        valor = float(config["gamma"] if gamma is None else gamma)
        miembros = []
        for ventana_analoga, futuro, parecido in zip(
                analogos.windows, analogos.futures, analogos.similarities):
            ordenada, pendiente = member_map(ventana_analoga, presente, parecido,
                                             gamma=valor)
            proyectado = np.clip(futuro, *entrada)
            miembros.append(np.clip(ordenada + pendiente * proyectado, *salida))
        miembros = np.array(miembros)
        salidas[nombre] = (ensemble_quantiles(miembros, LEVELS), miembros)
    return salidas


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
    for borrado in almacen.limpiar():
        print(f"  (se borró la corrida anterior en {Path(borrado).name})")

    ## Junto a la configuración congelada se guardan los dos extremos de la
    ## familia, que es la comparación que sostiene la tesis y sale casi gratis
    ## porque comparten la búsqueda de análogos. Si el gamma elegido coincide con
    ## un extremo, esa variante se omite en vez de duplicarla.
    variantes = {"Analog": None, "Analog-sin-contraccion": 0.0,
                 "Analog-minimos-cuadrados": 1.0}
    elegido = float(config["gamma"])
    for nombre, gamma in list(variantes.items()):
        if gamma is not None and gamma == elegido:
            del variantes[nombre]
            print(f"  (el gamma elegido es {elegido}: se omite {nombre}, "
                  f"que sería el mismo método)")

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
            reloj = time.time()
            try:
                salidas = ensambles_por_gamma(historia, config, variantes)
            except ValueError:
                continue
            ## el reloj se reparte entre las variantes: comparten la búsqueda, que
            ## es casi todo lo que cuesta
            por_variante = (time.time() - reloj) / len(salidas)
            for nombre, (cuantiles, miembros) in salidas.items():
                almacen.add(zona, origen, nombre, cuantiles, observado,
                            miembros, por_variante)
        parte = almacen.flush(zona)
        print(f"  {numero}/{len(zonas)} {zona:20s} "
              f"{time.time() - comienzo:6.0f}s  {parte.get('filas', 0):,d} filas",
              flush=True)

    print(f"\nescrito en {destino.with_suffix('')}")


if __name__ == "__main__":
    main()
