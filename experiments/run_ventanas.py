"""
El análogo con una sola ventana de búsqueda, de 24, 36 y 72 horas, con Pearson y con
distancia euclidiana.

La ventana de 48 horas se eligió en la malla de selección (24, 48, 72 y 168), donde 24
salió peor y 72 empató con 48. Aquí se prueba en el tramo de prueba, y se agrega 36 horas,
que la malla no tenía. Todo lo demás es la configuración de ``Analog``: 40 análogos,
separación de 0.5 veces la ventana, mapa con exponente 0 y recorte. La ventana de 48 ya
está corrida con las dos métricas (``Analog`` y ``Analog-euc``).
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

VENTANAS = (24, 36, 72)
METRICAS = {"pearson": "", "euclidiana": "-euc"}


def una_zona(tarea: tuple) -> dict:
    zona, config, destino = tarea
    comienzo = time.time()
    salida = Path(destino)
    if (salida.with_suffix("") / f"{zona}.parquet").exists():
        return {"zona": zona, "filas": 0, "segundos": 0.0}
    serie = load_series("pml", zona)
    almacen = ForecastStore(LEVELS, salida)
    k, sep, gamma = int(config["k"]), float(config["separation"]), float(config["gamma"])
    for origen in build_origins(TEST_START, TEST_END, serie):
        historia = serie.loc[:origen].values[-SEARCH_YEARS * 8760:]
        observado = serie.loc[origen + pd.Timedelta(hours=1):
                              origen + pd.Timedelta(hours=HORIZON)].values
        if len(observado) != HORIZON:
            continue
        recorte = limites(historia)
        for w in VENTANAS:
            for metrica, sufijo in METRICAS.items():
                reloj = time.time()
                try:
                    a = find_analogs(historia, w, HORIZON, k, sep, metric=metrica)
                except ValueError:
                    continue
                miembros = miembros_de(a, historia[len(historia) - w:], "normal", gamma,
                                       recorte)
                if len(miembros) >= 2:
                    almacen.add(zona, origen, f"Analog-w{w}{sufijo}",
                                ensemble_quantiles(miembros, LEVELS), observado,
                                miembros, time.time() - reloj)
    parte = almacen.flush(zona)
    return {"zona": zona, "filas": parte.get("filas", 0),
            "segundos": time.time() - comienzo}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--procesos", type=int, default=5)
    args = p.parse_args()
    config = load_selected()
    zonas = load_selected_zones()["zonas"]
    destino = RESULTS / "ventanas.parquet"
    comienzo = time.time()
    with Pool(args.procesos) as piscina:
        for n, h in enumerate(piscina.imap_unordered(
                una_zona, [(z, config, str(destino)) for z in zonas]), 1):
            print(f"  {n}/{len(zonas)} {h['zona']:22s} {h['segundos']:6.0f}s  "
                  f"(total {time.time() - comienzo:6.0f}s)", flush=True)
    print(f"\nescrito en {destino.with_suffix('')}")


if __name__ == "__main__":
    main()
