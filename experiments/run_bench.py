"""
Produce los pronósticos de los comparativos y los guarda. No califica nada.

Mismo contrato que `run_forecasts.py` para el análogo: mismos orígenes, mismo horizonte,
mismo almacén. Así la comparación no mezcla corridas y la capa de calibración se puede
aplicar después a todo el banco con las mismas reglas, que es lo que vuelve la
comparación honesta en vez de amañada a favor del método propio.

Cada comparativo entrega **trayectorias**, no cuantiles sueltos. Los ingenuos construyen
sus intervalos con sus errores pasados, de modo que el centro más cada vector de error
pasado ya es una muestra de caminos; emitirla así cuesta lo mismo y gana dos cosas: los
puntajes de trayectoria se pueden calcular para todo el banco, y todos reciben el mismo
tratamiento de calibración —escalado de miembros— en vez de uno distinto por método.
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

from analog_probabilistic.analog_probabilistic import ensemble_quantiles  # noqa: E402
from experiments.benchmarks.lear import LEARRodante  # noqa: E402
from experiments.benchmarks.naive import (naive, naive_combinado,  # noqa: E402
                                          naive_semanal)
from experiments.forecast_store import ForecastStore  # noqa: E402
from experiments.protocol import (HORIZON, RESULTS, SEARCH_YEARS,  # noqa: E402
                                  SELECTION_END, SELECTION_START, TEST_END,
                                  TEST_START, build_origins,
                                  load_selected_zones)
from experiments.scores import LEVELS  # noqa: E402
from experiments.series import load_series  # noqa: E402

TRAMOS = {"seleccion": (SELECTION_START, SELECTION_END),
          "prueba": (TEST_START, TEST_END)}

## Los que sólo necesitan la historia y devuelven trayectorias de una vez.
SIN_ESTADO = {
    "Naive-diario": lambda h: naive(h, HORIZON, LEVELS, miembros=True),
    "Naive-semanal": lambda h: naive_semanal(h, HORIZON, LEVELS, miembros=True),
    "Naive-combinado": lambda h: naive_combinado(h, HORIZON, LEVELS, miembros=True),
}
## Los que necesitan arrastrar errores de un origen al siguiente.
CON_ESTADO = ("LEAR",)
METODOS = tuple(SIN_ESTADO) + CON_ESTADO


def una_zona(tarea: tuple) -> dict:
    zona, inicio, fin, metodos, destino, refit = tarea
    comienzo = time.time()
    salida = Path(destino)
    if (salida.with_suffix("") / f"{zona}.parquet").exists():
        return {"zona": zona, "filas": 0, "segundos": 0.0, "reusada": True}

    serie = load_series("pml", zona)
    almacen = ForecastStore(LEVELS, salida)
    lear = LEARRodante() if "LEAR" in metodos else None
    ## LEAR se reajusta cada `refit` días y entre reajustes reusa sus coeficientes:
    ## a diario son 26 segundos por origen, trescientas veinticuatro horas para el
    ## panel entero, y eso no cabe. La desviación se declara en el artículo.
    ultimo_ajuste, coeficientes = None, None

    for numero, origen in enumerate(build_origins(inicio, fin, serie)):
        historia = serie.loc[:origen].values[-SEARCH_YEARS * 8760:]
        observado = serie.loc[origen + pd.Timedelta(hours=1):
                              origen + pd.Timedelta(hours=HORIZON)].values
        if len(observado) != HORIZON:
            continue

        for nombre in metodos:
            reloj = time.time()
            try:
                if nombre in SIN_ESTADO:
                    miembros = SIN_ESTADO[nombre](historia)
                else:
                    cuantiles = lear.predice(historia, HORIZON, LEVELS,
                                             dia_semana_origen=origen.weekday())
                    ## las trayectorias de LEAR son su centro más cada error pasado
                    miembros = lear.trayectorias()
                    lear.observa(observado)
                    if miembros is None:
                        continue
            except (ValueError, IndexError):
                if nombre == "LEAR" and lear is not None:
                    lear._ultimo = None
                continue
            if miembros is None or len(miembros) < 2:
                continue
            almacen.add(zona, origen, nombre,
                        ensemble_quantiles(miembros, LEVELS), observado,
                        miembros, time.time() - reloj)

    parte = almacen.flush(zona)
    return {"zona": zona, "filas": parte.get("filas", 0),
            "segundos": time.time() - comienzo, "reusada": False}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tramo", choices=sorted(TRAMOS), default="prueba")
    p.add_argument("--metodos", nargs="+", default=list(SIN_ESTADO))
    p.add_argument("--procesos", type=int, default=6)
    p.add_argument("--zonas", type=int, default=None)
    p.add_argument("--refit", type=int, default=7)
    p.add_argument("--salida", default=None)
    args = p.parse_args()

    desconocidos = [m for m in args.metodos if m not in METODOS]
    if desconocidos:
        raise SystemExit(f"métodos desconocidos: {desconocidos}; hay {METODOS}")

    inicio, fin = TRAMOS[args.tramo]
    zonas = load_selected_zones()["zonas"][: args.zonas]
    destino = Path(args.salida or RESULTS / f"bench_{args.tramo}.parquet")
    print(f"tramo {args.tramo}: {inicio} a {fin}")
    print(f"{len(zonas)} zonas · métodos: {', '.join(args.metodos)}")
    print(f"{args.procesos} procesos · destino {destino.with_suffix('').name}",
          flush=True)

    comienzo = time.time()
    tareas = [(z, inicio, fin, args.metodos, str(destino), args.refit) for z in zonas]
    with Pool(args.procesos) as piscina:
        for n, hecho in enumerate(piscina.imap_unordered(una_zona, tareas), 1):
            print(f"  {n}/{len(zonas)} {hecho['zona']:20s} {hecho['segundos']:6.0f}s  "
                  f"{hecho['filas']:,d} filas"
                  f"{'  (ya estaba)' if hecho['reusada'] else ''}  "
                  f"(total {time.time() - comienzo:6.0f}s)", flush=True)
    print(f"\nescrito en {destino.with_suffix('')}")


if __name__ == "__main__":
    main()
