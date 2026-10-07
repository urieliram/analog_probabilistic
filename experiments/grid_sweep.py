"""
La malla completa: ventana x número de análogos x separación x exponente de contracción.

Esto existe porque la ventana de 48 horas, los 40 análogos y la separación de 0.5 nunca
se eligieron. Venían del experimento anterior, cableados en `gamma_sweep.py`, y el
barrido de este rediseño sólo recorrió el exponente de contracción. La malla estaba
declarada en el protocolo y sin correr, de modo que la ventana no tenía detrás ninguna
comparación contra 24, 72 o 168 horas sobre estos datos.

Dos cosas que la malla va a decidir y que no son independientes entre sí:

1. **Ventana y número de análogos están atados.** La regla de separación limita cuántos
   análogos caben: con ventana de 168 horas y separación 1.0 caben 104 en dos años, de
   modo que pedir 100 es pedir casi todas las semanas del banco y el ensamble deja de ser
   de análogos para volverse una muestra del clima. Por eso se registra `k_real`, el
   número que de verdad se encontró, y no sólo el que se pidió.
2. **La contracción depende de la ventana.** El parecido medio de los análogos elegidos
   baja de 0.90 con ventana de 24 horas a 0.66 con 168, y la contracción multiplica la
   pendiente por ese parecido elevado a gamma. Si el argumento del artículo es correcto,
   el daño de contraer tiene que crecer con la ventana. Eso es una predicción con dosis y
   respuesta, y aquí se pone a prueba.

Se corre sobre dos ventanas de selección, no una. La primera, 2020-04 a 2021-03, son los
doce meses del valle de la pandemia: la mitad del nivel de precio y el 43% del
movimiento diario del tramo donde la configuración se aplica. No se eligió por descuido
—la serie empieza en abril de 2018 y el banco es de dos años, así que es el primer año
con banco completo— pero sí vuelve sospechosa cualquier elección hecha ahí. La segunda,
2022-04 a 2023-03, tiene el banco ya limpio de valle. Si las dos eligen lo mismo, la
objeción queda contestada con evidencia propia.
"""

import argparse
import itertools
import os
import sys
import time

## Un hilo de álgebra por proceso. Tiene que quedar antes de importar numpy, que es
## cuando se leen estas variables: con los hilos libres, seis procesos sobre ocho
## núcleos se estorban y la aceleración cayó de seis veces a menos de dos — 28 minutos
## por zona en vez de los nueve que tarda una sola.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_probabilistic.analog_probabilistic import (  # noqa: E402
    ensemble_quantiles, find_analogs, member_map)
from experiments.protocol import (GAMMA_GRID, HORIZON, K_GRID,  # noqa: E402
                                  RESULTS, SEARCH_YEARS, SELECTION_END,
                                  SELECTION_START, SEPARATION_GRID, WINDOW_GRID,
                                  build_origins, load_selected_zones)
from experiments.scores import LEVELS, evaluate  # noqa: E402
from experiments.series import load_series  # noqa: E402

## La segunda ventana de selección: mismo largo, banco sin valle de pandemia.
TRAMOS = {
    "valle": (SELECTION_START, SELECTION_END),
    "limpio": ("2022-04-01", "2023-03-31"),
}

## De los puntajes se guardan los que el artículo usa. Las diecinueve tasas de acierto
## no caben —tres millones de filas por diecinueve columnas— y `reliability` las resume;
## se conservan las dos colas, que son donde vive el argumento.
COLUMNAS = ("mean_pinball", "crps", "reliability", "mae", "rmse",
            "cover80", "sharp80", "winkler80", "cover90", "sharp90", "winkler90",
            "pinball95", "wis", "hit_0.05", "hit_0.95")

CELDAS = tuple(itertools.product(WINDOW_GRID, K_GRID, SEPARATION_GRID))


def una_zona(tarea: tuple) -> dict:
    """Recorre la malla completa sobre todos los orígenes de una zona."""
    zona, inicio, fin = tarea
    comienzo = time.time()

    ## una zona ya escrita no se repite: la corrida son horas y conviene poder
    ## interrumpirla y seguir donde quedó
    destino = RESULTS / f"grid_{inicio[:7]}"
    salida = destino / f"{zona}.parquet"
    if salida.exists():
        return {"zona": zona, "filas": len(pd.read_parquet(salida)),
                "segundos": 0.0, "reusada": True}

    serie = load_series("pml", zona)
    filas = []

    for origen in build_origins(inicio, fin, serie):
        historia = serie.loc[:origen].values[-SEARCH_YEARS * 8760:]
        observado = serie.loc[origen + pd.Timedelta(hours=1):
                              origen + pd.Timedelta(hours=HORIZON)].values
        if len(observado) != HORIZON:
            continue

        for ventana, k, separacion in CELDAS:
            try:
                analogos = find_analogs(historia, ventana, HORIZON, k, separacion)
            except ValueError:
                continue
            presente = historia[len(historia) - ventana:]
            ## la búsqueda se comparte entre los once exponentes: es casi todo el costo
            for gamma in GAMMA_GRID:
                miembros = np.array([
                    ordenada + pendiente * futuro
                    for (ordenada, pendiente), futuro in (
                        (member_map(w, presente, s, gamma=gamma), f)
                        for w, f, s in zip(analogos.windows, analogos.futures,
                                           analogos.similarities))])
                puntajes = evaluate(observado, ensemble_quantiles(miembros, LEVELS),
                                    LEVELS, miembros)
                fila = {"zona": zona, "origen": origen, "window": ventana, "k": k,
                        "separation": separacion, "gamma": gamma,
                        "k_real": len(analogos.similarities),
                        "parecido_medio": float(np.mean(analogos.similarities)),
                        "parecido_peor": float(np.min(analogos.similarities))}
                fila.update({c: puntajes[c] for c in COLUMNAS if c in puntajes})
                filas.append(fila)

    tabla = pd.DataFrame(filas)
    destino.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(salida, index=False)
    return {"zona": zona, "filas": len(tabla),
            "segundos": time.time() - comienzo, "reusada": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tramo", choices=sorted(TRAMOS), default="valle")
    parser.add_argument("--procesos", type=int, default=6)
    parser.add_argument("--zonas", type=int, default=None)
    args = parser.parse_args()

    inicio, fin = TRAMOS[args.tramo]
    zonas = load_selected_zones()["zonas"][: args.zonas]
    print(f"tramo {args.tramo}: {inicio} a {fin}")
    print(f"{len(zonas)} zonas x {len(CELDAS)} celdas x {len(GAMMA_GRID)} exponentes")
    print(f"{args.procesos} procesos", flush=True)

    comienzo = time.time()
    tareas = [(z, inicio, fin) for z in zonas]
    with Pool(args.procesos) as piscina:
        for numero, hecho in enumerate(piscina.imap_unordered(una_zona, tareas), 1):
            print(f"  {numero}/{len(zonas)} {hecho['zona']:20s} "
                  f"{hecho['segundos']:5.0f}s  {hecho['filas']:,d} filas"
                  f"{'  (ya estaba)' if hecho['reusada'] else ''}  "
                  f"(total {time.time() - comienzo:5.0f}s)", flush=True)

    print(f"\nescrito en {RESULTS / f'grid_{inicio[:7]}'}")


if __name__ == "__main__":
    main()
