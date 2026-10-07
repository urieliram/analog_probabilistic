"""
Variantes del análogo pensadas para que el abanico alcance los picos.

El diagnóstico que las motiva está medido: la amplitud del abanico sale enteramente de
la desviación del presente, porque la pendiente del mapa es
sd(presente) / sd(ventana análoga). Si la noche anterior estuvo tranquila la amplitud es
chica, **sin importar lo que hicieron las continuaciones de los análogos**, y el día
siguiente puede ser movido. En el cuarto de días más movidos el precio se escapa por
arriba el 10.4% de las veces cuando se prometió 5%, y la calibración conformal sólo
lo baja a 6.2% a cambio de 53% más de ancho en todas partes.

Cinco variantes, y cada una ataca el diagnóstico por un punto distinto:

``Analog``              la configuración congelada, como testigo: ventana 48, k 40,
                        separación 0.5, exponente 0, con recorte.
``sin-recorte``         el recorte de salida tiene un techo en el máximo histórico, de
                        modo que el método NO PUEDE anunciar un precio récord. Eso
                        censura la cola derecha por construcción. En promedio mueve el
                        0.2% de los miembros, pero hay ensambles donde pega tantos
                        miembros al techo que el cuantil 95 coincide con la mediana —
                        salió al acotar el factor conformal, en el 0.46% de los pasos.
                        Aquí se apaga.
``crudo``               pendiente 1: cada continuación conserva su propia amplitud y
                        sólo se recorre de nivel. Es la prueba directa de la hipótesis:
                        si una continuación fue un pico, que siga siendo un pico.
``amplitud-libre``      pendiente max(1, sd(presente)/sd(análogo)): se permite
                        agrandar la amplitud pero nunca encogerla. Un análogo movido
                        conserva su movimiento aunque el presente esté tranquilo.
``mezcla``              junta los miembros de las cuatro ventanas, 24, 48, 72 y 168
                        horas, con k 40 cada una. Las cuatro miran escalas distintas de
                        memoria, y el abanico se ensancha **más cuando no se ponen de
                        acuerdo entre sí**, que es señal de que el régimen está
                        cambiando. Es la idea del usuario, y es la única de las cinco
                        cuyo ancho responde a algo del futuro y no sólo del presente.

Todas comparten la búsqueda de análogos de su ventana, que es casi todo el costo.
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
                                  TEST_END, TEST_START, WINDOW_GRID,
                                  build_origins, load_selected,
                                  load_selected_zones)
from experiments.scores import LEVELS  # noqa: E402
from experiments.series import load_series  # noqa: E402

VENTANAS_MEZCLA = WINDOW_GRID


def limites(historia: np.ndarray) -> tuple:
    return (float(historia.min()), float(historia.max()),
            float(np.percentile(historia, 1)) - 2.0 * float(historia.std()),
            float(historia.max()))


def miembros_de(analogos, presente: np.ndarray, modo: str, gamma: float,
                recorte: tuple) -> np.ndarray:
    """Arma el ensamble con la pendiente que pide cada variante."""
    ent_baja, ent_alta, sal_baja, sal_alta = recorte
    sd_presente = float(np.std(presente))
    media_presente = float(np.mean(presente))

    salida = []
    for ventana, futuro, parecido in zip(analogos.windows, analogos.futures,
                                         analogos.similarities):
        sd_analogo = float(np.std(ventana))
        if sd_analogo == 0:
            pendiente = 1.0
        elif modo == "crudo":
            pendiente = 1.0
        elif modo == "amplitud-libre":
            pendiente = max(1.0, sd_presente / sd_analogo)
        else:
            pendiente = sd_presente / sd_analogo
        if gamma:
            pendiente *= float(parecido) ** gamma
        ordenada = media_presente - pendiente * float(np.mean(ventana))

        if recorte[0] is None:
            salida.append(ordenada + pendiente * futuro)
        else:
            proyectado = np.clip(futuro, ent_baja, ent_alta)
            salida.append(np.clip(ordenada + pendiente * proyectado,
                                  sal_baja, sal_alta))
    return np.array(salida)


def una_zona(tarea: tuple) -> dict:
    zona, inicio, fin, config, destino = tarea
    comienzo = time.time()
    salida = Path(destino)
    if (salida.with_suffix("") / f"{zona}.parquet").exists():
        return {"zona": zona, "filas": 0, "segundos": 0.0, "reusada": True}

    serie = load_series("pml", zona)
    almacen = ForecastStore(LEVELS, salida)
    ventana, k = int(config["window"]), int(config["k"])
    sep, gamma = float(config["separation"]), float(config["gamma"])

    for origen in build_origins(inicio, fin, serie):
        historia = serie.loc[:origen].values[-SEARCH_YEARS * 8760:]
        observado = serie.loc[origen + pd.Timedelta(hours=1):
                              origen + pd.Timedelta(hours=HORIZON)].values
        if len(observado) != HORIZON:
            continue
        recorte = limites(historia)
        sin_recorte = (None, None, None, None)

        reloj = time.time()
        try:
            analogos = find_analogs(historia, ventana, HORIZON, k, sep)
        except ValueError:
            continue
        presente = historia[len(historia) - ventana:]
        costo_busqueda = time.time() - reloj

        ensambles = {
            "Analog": miembros_de(analogos, presente, "normal", gamma, recorte),
            "Analog-sin-recorte": miembros_de(analogos, presente, "normal", gamma,
                                              sin_recorte),
            "Analog-crudo": miembros_de(analogos, presente, "crudo", gamma, recorte),
            "Analog-amplitud-libre": miembros_de(analogos, presente,
                                                 "amplitud-libre", gamma, recorte),
        }

        ## la mezcla necesita su propia búsqueda por ventana
        reloj = time.time()
        partes = []
        for w in VENTANAS_MEZCLA:
            try:
                a = find_analogs(historia, w, HORIZON, k, sep)
            except ValueError:
                continue
            partes.append(miembros_de(a, historia[len(historia) - w:], "normal",
                                      gamma, recorte))
        if partes:
            ensambles["Analog-mezcla"] = np.vstack(partes)
        costo_mezcla = time.time() - reloj

        for nombre, miembros in ensambles.items():
            if len(miembros) < 2:
                continue
            costo = (costo_busqueda + costo_mezcla if nombre == "Analog-mezcla"
                     else costo_busqueda / 4)
            almacen.add(zona, origen, nombre,
                        ensemble_quantiles(miembros, LEVELS), observado,
                        miembros, costo)

    parte = almacen.flush(zona)
    return {"zona": zona, "filas": parte.get("filas", 0),
            "segundos": time.time() - comienzo, "reusada": False}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--procesos", type=int, default=8)
    p.add_argument("--zonas", type=int, default=None)
    p.add_argument("--salida", default=None)
    args = p.parse_args()

    config = load_selected()
    zonas = load_selected_zones()["zonas"][: args.zonas]
    destino = Path(args.salida or RESULTS / "picos_prueba.parquet")
    print(f"tramo de prueba: {TEST_START} a {TEST_END}")
    print(f"{len(zonas)} zonas · 5 variantes · {args.procesos} procesos", flush=True)

    comienzo = time.time()
    tareas = [(z, TEST_START, TEST_END, config, str(destino)) for z in zonas]
    with Pool(args.procesos) as piscina:
        for n, h in enumerate(piscina.imap_unordered(una_zona, tareas), 1):
            print(f"  {n}/{len(zonas)} {h['zona']:22s} {h['segundos']:6.0f}s  "
                  f"{h['filas']:,d} filas{'  (ya estaba)' if h['reusada'] else ''}  "
                  f"(total {time.time() - comienzo:6.0f}s)", flush=True)
    print(f"\nescrito en {destino.with_suffix('')}")


if __name__ == "__main__":
    main()
