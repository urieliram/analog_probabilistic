"""
Analog-memoria-larga: la mezcla de ventanas, con más memoria y más diversidad.

Analog-mezcla junta cuatro ventanas de búsqueda (24, 48, 72 y 168 horas). Aquí se juntan
doce: 24, 36, 48 y 72, y de 72 a 168 cada 12 horas (84, 96, ..., 168), con 40 análogos
cada una, para 480 escenarios. La idea, del usuario: que la búsqueda no sea miope a dos
días, sino que encuentre regímenes de varios días —tres días caros seguidos, o cuatro
caros y uno barato— y proyecte lo que les siguió.

Advertencia que hay que leer junto con los resultados: la búsqueda es por correlación
de Pearson, que mide la forma y no el nivel. Una ventana larga distingue tres días que
suben de tres días que bajan, pero no tres días caros de tres días baratos con la misma
forma. El nivel lo pone después el mapa lineal, con la media del presente.

Todo lo demás es idéntico a Analog-mezcla (``run_picos.py``): mismo mapa (exponente 0),
mismo recorte, misma separación de 0.5 veces la ventana.
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
                                  TEST_END, TEST_START, build_origins,
                                  load_selected, load_selected_zones)
from experiments.run_picos import limites, miembros_de  # noqa: E402
from experiments.scores import LEVELS  # noqa: E402
from experiments.series import load_series  # noqa: E402

VENTANAS = (24, 36, 48, 72) + tuple(range(84, 169, 12))
NOMBRE = "Analog-memoria-larga"


def una_zona(tarea: tuple) -> dict:
    zona, inicio, fin, config, destino, limite = tarea
    comienzo = time.time()
    salida = Path(destino)
    if (salida.with_suffix("") / f"{zona}.parquet").exists():
        return {"zona": zona, "filas": 0, "segundos": 0.0, "reusada": True}
    serie = load_series("pml", zona)
    almacen = ForecastStore(LEVELS, salida)
    k, sep, gamma = int(config["k"]), float(config["separation"]), float(config["gamma"])
    origenes = build_origins(inicio, fin, serie)
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
        partes = []
        for w in VENTANAS:
            try:
                a = find_analogs(historia, w, HORIZON, k, sep)
            except ValueError:
                continue
            partes.append(miembros_de(a, historia[len(historia) - w:], "normal",
                                      gamma, recorte))
        if not partes:
            continue
        miembros = np.vstack(partes)
        almacen.add(zona, origen, NOMBRE, ensemble_quantiles(miembros, LEVELS),
                    observado, miembros, time.time() - reloj)
    parte = almacen.flush(zona)
    return {"zona": zona, "filas": parte.get("filas", 0),
            "segundos": time.time() - comienzo, "reusada": False}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--procesos", type=int, default=5)
    p.add_argument("--zonas", type=int, default=None)
    p.add_argument("--origenes", type=int, default=None)
    p.add_argument("--salida", default=None)
    args = p.parse_args()
    config = load_selected()
    zonas = load_selected_zones()["zonas"][: args.zonas]
    destino = Path(args.salida or RESULTS / "memoria_larga.parquet")
    print(f"{NOMBRE}: ventanas {VENTANAS}, {len(VENTANAS) * int(config['k'])} escenarios; "
          f"{len(zonas)} zonas, {args.procesos} procesos", flush=True)
    comienzo = time.time()
    tareas = [(z, TEST_START, TEST_END, config, str(destino), args.origenes) for z in zonas]
    with Pool(args.procesos) as piscina:
        for n, h in enumerate(piscina.imap_unordered(una_zona, tareas), 1):
            print(f"  {n}/{len(zonas)} {h['zona']:22s} {h['segundos']:6.0f}s  "
                  f"{h['filas']:,d} filas  (total {time.time() - comienzo:6.0f}s)",
                  flush=True)
    print(f"\nescrito en {destino.with_suffix('')}")


if __name__ == "__main__":
    main()
