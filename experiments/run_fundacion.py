"""
Pronósticos de los modelos de fundación sobre el tramo de prueba, al almacén común.

Mismo contrato que el análogo y que los demás comparativos: mismos orígenes, mismo
horizonte, mismo almacén, para que la capa de calibración se aplique después a todo el
banco con las mismas reglas.

Tres cosas que hay que saber para leer sus números, y las tres se midieron, no se
supusieron:

1. **Ninguno llega a los mismos niveles de cuantil.** TiRex y TimesFM se detienen en 0.9,
   así que su intervalo central más ancho de fábrica es el de 80%; t0-beta llega a 0.99 y
   PatchTST-FM a 0.99 de centésima en centésima. Los niveles que un modelo no emite se
   guardan como ausentes, nunca interpolados: que un nivel no exista es un hecho del
   modelo y el artículo lo reporta. La comparación entre métodos se hace sobre los nueve
   niveles que TODOS emiten, de 0.1 a 0.9.

2. **Ninguno entrega trayectorias.** Moirai parecía entregarlas —mil muestras por
   pronóstico— pero la correlación entre horas vecinas dentro de un mismo camino es
   +0.005, estadísticamente cero, contra +0.644 del análogo y −0.043 de un sorteo
   deliberadamente independiente. Es decir que cada muestra es un sorteo del producto de
   las 24 distribuciones por hora, no un día posible. Se guardan igual, porque medir eso
   es un resultado, pero no son miembros coherentes.

3. **Los pesos de Moirai son no comerciales y los de TimesFM 3.0 además no distribuibles**,
   así que ninguno se copia dentro de este repositorio; los adaptadores los leen de la
   caché local.
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.forecast_store import ForecastStore  # noqa: E402
from experiments.protocol import (HORIZON, RESULTS, SEARCH_YEARS,  # noqa: E402
                                  TEST_END, TEST_START, build_origins,
                                  load_selected_zones)
from experiments.scores import LEVELS  # noqa: E402
from experiments.series import load_series  # noqa: E402

## nombre del método en el cuadro -> módulo del adaptador, en orden de lo barato a lo
## caro según lo medido en esta tarjeta
ADAPTADORES = (
    ("t0-beta", "fundacion_t0"),
    ("PatchTST-FM", "fundacion_patchtst"),
    ("TimesFM-2.5", "fundacion_timesfm"),
    ("TiRex", "fundacion_tirex"),
    ("Moirai", "fundacion_moirai"),
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modelo", required=True,
                   choices=[n for n, _ in ADAPTADORES])
    p.add_argument("--zonas", type=int, default=None)
    p.add_argument("--origenes", type=int, default=None,
                   help="limita los orígenes, para pruebas rápidas")
    p.add_argument("--dispositivo", default="cuda")
    args = p.parse_args()

    modulo = dict(ADAPTADORES)[args.modelo]
    adaptador = importlib.import_module(f"experiments.benchmarks.{modulo}")
    nativos = set(np.round(np.asarray(adaptador.NIVELES_NATIVOS, dtype=float), 4))
    faltantes = sorted(set(np.round(LEVELS, 4)) - nativos)

    zonas = load_selected_zones()["zonas"][: args.zonas]
    destino = RESULTS / f"fundacion_{args.modelo}.parquet"
    print(f"{args.modelo}: {len(zonas)} zonas, tramo {TEST_START} a {TEST_END}")
    print(f"  niveles nativos: {len(nativos)}; de los {len(LEVELS)} del banco "
          f"faltan {len(faltantes)}: {faltantes}")
    print(f"  cargando en {args.dispositivo}...", flush=True)

    reloj = time.time()
    modelo = adaptador.carga(args.dispositivo)
    print(f"  cargado en {time.time() - reloj:.1f}s", flush=True)

    almacen = ForecastStore(LEVELS, destino)
    comienzo = time.time()
    for numero, zona in enumerate(zonas, 1):
        if (destino.with_suffix("") / f"{zona}.parquet").exists():
            print(f"  {numero}/{len(zonas)} {zona:20s} (ya estaba)", flush=True)
            continue
        serie = load_series("pml", zona)
        origenes = build_origins(TEST_START, TEST_END, serie)
        if args.origenes:
            origenes = origenes[: args.origenes]
        fallos = 0
        for origen in origenes:
            historia = serie.loc[:origen].values[-SEARCH_YEARS * 8760:]
            observado = serie.loc[origen + pd.Timedelta(hours=1):
                                  origen + pd.Timedelta(hours=HORIZON)].values
            if len(observado) != HORIZON:
                continue
            t = time.time()
            try:
                cuantiles, miembros = adaptador.predice(modelo, historia, HORIZON,
                                                        list(LEVELS))
            except Exception:
                fallos += 1
                continue
            almacen.add(zona, origen, args.modelo, np.asarray(cuantiles, dtype=float),
                        observado, miembros, time.time() - t)
        parte = almacen.flush(zona)
        aviso = f"  ({fallos} fallos)" if fallos else ""
        print(f"  {numero}/{len(zonas)} {zona:20s} {parte.get('filas', 0):,d} filas"
              f"{aviso}  (total {time.time() - comienzo:6.0f}s)", flush=True)

    print(f"\nescrito en {destino.with_suffix('')}")


if __name__ == "__main__":
    main()
