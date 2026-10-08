"""
Todas las variantes del análogo, con distancia euclidiana en lugar de Pearson.

Pearson mide sólo la forma de la ventana; la distancia euclidiana, sobre los precios
crudos, mide forma y nivel. Con Pearson, tres días caros y tres días baratos de la
misma forma son igual de parecidos al presente; con euclidiana, no. Se prueba si buscar
por nivel ayuda a anticipar regímenes caros.

Las seis variantes son las mismas de ``run_picos.py`` y ``run_memoria_larga.py``, con
idéntico mapa, recorte, separación y número de análogos; sólo cambia la métrica. Los
nombres llevan el sufijo ``-euc``.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_probabilistic.analog_probabilistic import (  # noqa: E402
    ensemble_quantiles, find_analogs)
from experiments.forecast_store import ForecastStore  # noqa: E402
from experiments.protocol import (HORIZON, RESULTS, SEARCH_YEARS,  # noqa: E402
                                  TEST_END, TEST_START, WINDOW_GRID, build_origins,
                                  load_selected, load_selected_zones)
from experiments.run_memoria_larga import VENTANAS as VENTANAS_LARGA  # noqa: E402
from experiments.run_picos import limites, miembros_de  # noqa: E402
from experiments.scores import LEVELS  # noqa: E402
from experiments.series import load_series  # noqa: E402

METRICA = "euclidiana"


def una_zona(tarea: tuple) -> dict:
    zona, config, destino, limite = tarea
    comienzo = time.time()
    salida = Path(destino)
    if (salida.with_suffix("") / f"{zona}.parquet").exists():
        return {"zona": zona, "filas": 0, "segundos": 0.0}
    serie = load_series("pml", zona)
    almacen = ForecastStore(LEVELS, salida)
    w, k = int(config["window"]), int(config["k"])
    sep, gamma = float(config["separation"]), float(config["gamma"])
    origenes = build_origins(TEST_START, TEST_END, serie)
    if limite:
        origenes = origenes[:limite]
    for origen in origenes:
        historia = serie.loc[:origen].values[-SEARCH_YEARS * 8760:]
        observado = serie.loc[origen + pd.Timedelta(hours=1):
                              origen + pd.Timedelta(hours=HORIZON)].values
        if len(observado) != HORIZON:
            continue
        recorte = limites(historia)
        reloj = time.time()
        busquedas = {}
        for v in sorted(set(WINDOW_GRID) | set(VENTANAS_LARGA) | {w}):
            try:
                busquedas[v] = find_analogs(historia, v, HORIZON, k, sep, metric=METRICA)
            except ValueError:
                pass
        if w not in busquedas:
            continue
        a, presente = busquedas[w], historia[len(historia) - w:]
        ens = {
            "Analog-euc": miembros_de(a, presente, "normal", gamma, recorte),
            "Analog-sin-recorte-euc": miembros_de(a, presente, "normal", gamma,
                                                  (None, None, None, None)),
            "Analog-crudo-euc": miembros_de(a, presente, "crudo", gamma, recorte),
            "Analog-amplitud-libre-euc": miembros_de(a, presente, "amplitud-libre",
                                                     gamma, recorte),
        }
        for nombre, ventanas in [("Analog-mezcla-euc", WINDOW_GRID),
                                 ("Analog-memoria-larga-euc", VENTANAS_LARGA)]:
            partes = [miembros_de(busquedas[v], historia[len(historia) - v:], "normal",
                                  gamma, recorte) for v in ventanas if v in busquedas]
            if partes:
                ens[nombre] = np.vstack(partes)
        costo = time.time() - reloj
        for nombre, miembros in ens.items():
            if len(miembros) >= 2:
                almacen.add(zona, origen, nombre, ensemble_quantiles(miembros, LEVELS),
                            observado, miembros, costo)
    parte = almacen.flush(zona)
    return {"zona": zona, "filas": parte.get("filas", 0),
            "segundos": time.time() - comienzo}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--procesos", type=int, default=5)
    p.add_argument("--zonas", type=int, default=None)
    p.add_argument("--origenes", type=int, default=None)
    p.add_argument("--salida", default=None)
    args = p.parse_args()
    config = load_selected()
    zonas = load_selected_zones()["zonas"][: args.zonas]
    destino = Path(args.salida or RESULTS / "euclidiana.parquet")
    print(f"análogo con distancia euclidiana: 6 variantes, {len(zonas)} zonas, "
          f"{args.procesos} procesos", flush=True)
    comienzo = time.time()
    tareas = [(z, config, str(destino), args.origenes) for z in zonas]
    with Pool(args.procesos) as piscina:
        for n, h in enumerate(piscina.imap_unordered(una_zona, tareas), 1):
            print(f"  {n}/{len(zonas)} {h['zona']:22s} {h['segundos']:6.0f}s  "
                  f"{h['filas']:,d} filas  (total {time.time() - comienzo:6.0f}s)",
                  flush=True)
    print(f"\nescrito en {destino.with_suffix('')}")


if __name__ == "__main__":
    main()
