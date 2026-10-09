"""
Analog-mezcla con filtro de calendario: los análogos se buscan sólo entre continuaciones
del mismo tipo de día que el que se pronostica.

Diagnóstico que lo motiva (2026-10-08): la brecha del centro del análogo contra t0-beta es
de 4% de martes a viernes, 11% los lunes, 14% los sábados y 48% los domingos (208 contra
140 pesos de error absoluto medio). El análogo busca la forma de las últimas 48 horas y
sus continuaciones pueden ser de cualquier día: no sabe que mañana es domingo. LEAR, que
usa el día de la semana, acierta mucho más en domingo.

Es la separación de Altshuller: la categoría (tipo de día) se aplica como filtro y la
forma (Pearson) se busca dentro del grupo, en lugar de mezclar las dos en una distancia.

Dos filtros:
``cal3``  tres tipos: laborable, sábado, y domingo o festivo oficial (``holidays.MX``).
``cal7``  el mismo día de la semana, con los festivos oficiales tratados como domingo.

El tipo de día de una continuación candidata se lee en la hora central de la
continuación (posición + ventana + 12), porque los candidatos están a cualquier fase
horaria y la correlación ya los alinea casi siempre a la fase del presente. Todo lo demás
es igual a ``Analog-mezcla`` y a ``Analog``: mismas ventanas, k, separación, mapa y recorte.

Hipótesis declarada antes de correr (H8): el filtro baja el error absoluto medio del
centro de Analog-mezcla al menos 3%, con la ganancia concentrada en domingos, sábados y
lunes, y mejora la CRPS ponderada de la cuenta del día calibrada por nivel y bloque.

Segundo diagnóstico (medido antes de correr, 3 zonas, 4 ventanas): sólo el 41-43% de
los análogos elegidos empieza su continuación a la hora 0, el desfase medio es de 1.0 a
1.3 horas y entre el 9 y el 16% se corre más de dos horas. Una continuación que empieza
a las 19:00 se usa como si fuera el día de 0 a 23, y al promediar picos corridos el
centro se aplana. Por eso hay dos filtros más:
``fase``     sólo continuaciones alineadas: que empiecen a la hora 0 (cualquier día).
``cal7fase`` el mismo día de la semana y alineadas.

Hipótesis declarada antes de correr (H11): el filtro de fase baja el error del centro
sobre todo en las horas del pico de la tarde, y cal7fase es la mejor de las cuatro.

Métodos que produce: Analog-{cal3,cal7,fase,cal7fase} (ventana 48) y
Analog-mezcla-{cal3,cal7,fase,cal7fase} (cuatro ventanas, 160 escenarios).
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

import holidays
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_probabilistic.analog_probabilistic import (  # noqa: E402
    ensemble_quantiles, find_analogs)
from experiments.forecast_store import ForecastStore  # noqa: E402
from experiments.protocol import (HORIZON, RESULTS, SEARCH_YEARS,  # noqa: E402
                                  TEST_END, TEST_START, WINDOW_GRID, build_origins,
                                  load_selected, load_selected_zones)
from experiments.run_picos import limites, miembros_de  # noqa: E402
from experiments.scores import LEVELS  # noqa: E402
from experiments.series import load_series  # noqa: E402

FESTIVOS = holidays.MX(years=range(2016, 2028))


def tipo_de_dia(indice: pd.DatetimeIndex) -> tuple:
    """Devuelve (tipo de 3 clases, tipo de 7 clases) para cada marca de tiempo."""
    dow = indice.dayofweek.to_numpy()
    festivo = np.array([d in FESTIVOS for d in indice.date])
    siete = np.where(festivo, 6, dow)
    tres = np.where(siete == 6, 2, np.where(siete == 5, 1, 0))
    return tres, siete


def una_zona(tarea: tuple) -> dict:
    zona, config, destino, limite, inicio, fin = tarea
    comienzo = time.time()
    salida = Path(destino)
    if (salida.with_suffix("") / f"{zona}.parquet").exists():
        return {"zona": zona, "filas": 0, "segundos": 0.0}
    serie = load_series("pml", zona)
    almacen = ForecastStore(LEVELS, salida)
    k, sep, gamma = int(config["k"]), float(config["separation"]), float(config["gamma"])
    origenes = build_origins(inicio, fin, serie)
    if limite:
        origenes = origenes[:limite]
    for origen in origenes:
        pasado = serie.loc[:origen].iloc[-SEARCH_YEARS * 8760:]
        historia, indice = pasado.values, pasado.index
        observado = serie.loc[origen + pd.Timedelta(hours=1):
                              origen + pd.Timedelta(hours=HORIZON)].values
        if len(observado) != HORIZON:
            continue
        recorte = limites(historia)
        ## tipo del día pronosticado: la hora 12 del día siguiente al origen
        tres_s, siete_s = tipo_de_dia(indice)
        tres_o, siete_o = tipo_de_dia(pd.DatetimeIndex([origen + pd.Timedelta(hours=13)]))
        horas = indice.hour.to_numpy()
        tipos = {"cal3": (tres_s, tres_o[0], False), "cal7": (siete_s, siete_o[0], False),
                 "fase": (np.zeros(len(indice), dtype=int), 0, True),
                 "cal7fase": (siete_s, siete_o[0], True)}
        if inicio != TEST_START:
            ## fuera del tramo de prueba hace falta también el testigo sin filtro
            tipos["sinfiltro"] = (np.zeros(len(indice), dtype=int), 0, False)
        reloj = time.time()
        for sufijo, (vector, tipo_objetivo, alineado) in tipos.items():
            partes = []
            for w in WINDOW_GRID:
                last = len(historia) - 2 * w - HORIZON + 1
                if last <= 0:
                    continue
                ## tipo de día de la hora central de cada continuación candidata
                centro = np.arange(last) + w + 12
                admisibles = vector[centro] == tipo_objetivo
                if alineado:
                    ## la continuación empieza a la hora 0: alineada con el día
                    admisibles &= horas[np.arange(last) + w] == 0
                try:
                    a = find_analogs(historia, w, HORIZON, k, sep, admisibles=admisibles)
                except ValueError:
                    continue
                miembros = miembros_de(a, historia[len(historia) - w:], "normal", gamma,
                                       recorte)
                partes.append(miembros)
                if w == int(config["window"]) and len(miembros) >= 2:
                    almacen.add(zona, origen, f"Analog-{sufijo}".replace("-sinfiltro", ""),
                                ensemble_quantiles(miembros, LEVELS), observado,
                                miembros, time.time() - reloj)
            if partes:
                mezcla = np.vstack(partes)
                almacen.add(zona, origen, f"Analog-mezcla-{sufijo}".replace("-sinfiltro", ""),
                            ensemble_quantiles(mezcla, LEVELS), observado, mezcla,
                            time.time() - reloj)
    parte = almacen.flush(zona)
    return {"zona": zona, "filas": parte.get("filas", 0),
            "segundos": time.time() - comienzo}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--procesos", type=int, default=5)
    p.add_argument("--zonas", type=int, default=None)
    p.add_argument("--origenes", type=int, default=None)
    p.add_argument("--salida", default=None)
    p.add_argument("--inicio", default=TEST_START)
    p.add_argument("--fin", default=TEST_END)
    args = p.parse_args()
    config = load_selected()
    zonas = load_selected_zones()["zonas"][: args.zonas]
    destino = Path(args.salida or RESULTS / "calendario.parquet")
    print(f"análogo con filtro de calendario: {len(zonas)} zonas, {args.procesos} procesos",
          flush=True)
    comienzo = time.time()
    tareas = [(z, config, str(destino), args.origenes, args.inicio, args.fin) for z in zonas]
    with Pool(args.procesos) as piscina:
        for n, h in enumerate(piscina.imap_unordered(una_zona, tareas), 1):
            print(f"  {n}/{len(zonas)} {h['zona']:22s} {h['segundos']:6.0f}s  "
                  f"{h['filas']:,d} filas  (total {time.time() - comienzo:6.0f}s)",
                  flush=True)
    print(f"\nescrito en {destino.with_suffix('')}")


if __name__ == "__main__":
    main()
