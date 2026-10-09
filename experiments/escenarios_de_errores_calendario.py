"""
La receta de escenarios de errores, con el mismo filtro de calendario que se le dio al
análogo.

Trato igual: el análogo con filtro de calendario (``run_calendario.py``, variante cal7)
sólo toma continuaciones del mismo día de la semana que el que se pronostica, con los
festivos oficiales tratados como domingo. La receta de errores de los rivales usaba los
errores de los últimos sesenta días sin importar su tipo. Aquí se usan los errores de los
últimos sesenta días del mismo tipo:

    trayectoria_i = mediana de hoy + vector de error del día pasado i del mismo tipo

con los mismos siete tipos que cal7. Para un domingo, sesenta domingos son unos catorce
meses; al principio del tramo hay menos, y se usan los que haya. Nada mira más allá del
origen.

Hipótesis declarada antes de correr (H12): si el filtro de calendario ayuda al análogo
por el calendario y no por otra cosa, también debería ayudar algo a los escenarios de
errores de t0-beta, PatchTST-FM y LEAR; la comparación justa del análogo con calendario
es contra estas versiones.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.forecast_store import ForecastStore  # noqa: E402
from experiments.protocol import RESULTS  # noqa: E402
from experiments.run_calendario import tipo_de_dia  # noqa: E402
from experiments.scores import LEVELS  # noqa: E402

VENTANA_ERRORES = 60
FUENTES = [("fundacion_t0-beta", "t0-beta"), ("fundacion_PatchTST-FM", "PatchTST-FM"),
           ("lear_prueba", "LEAR")]


def main():
    destino = RESULTS / "err_cal.parquet"
    almacen = ForecastStore(LEVELS, destino)
    zonas = sorted(p.stem for p in (RESULTS / FUENTES[0][0]).glob("*.parquet"))
    for n, zona in enumerate(zonas, 1):
        for corrida, metodo in FUENTES:
            t = pd.read_parquet(RESULTS / corrida / f"{zona}.parquet",
                                columns=["zone", "origin", "method", "step", "q0.5",
                                         "observed"])
            t = t[t.method == metodo]
            errores = {k: [] for k in range(7)}
            for origen, p in sorted(t.groupby("origin"), key=lambda kv: kv[0]):
                p = p.sort_values("step")
                observado, mediana = p["observed"].to_numpy(), p["q0.5"].to_numpy()
                if np.isnan(mediana).any() or len(mediana) != 24:
                    continue
                _, siete = tipo_de_dia(pd.DatetimeIndex([origen + pd.Timedelta(hours=13)]))
                tipo = int(siete[0])
                propios = errores[tipo]
                if len(propios) >= 2:
                    miembros = mediana[None, :] + np.array(propios[-VENTANA_ERRORES:])
                    almacen.add(zona, origen, f"{metodo}+errores-cal7",
                                np.quantile(miembros, LEVELS, axis=0), observado,
                                miembros, float("nan"))
                propios.append(observado - mediana)
        almacen.flush(zona)
        print(f"  {n}/{len(zonas)} {zona}", flush=True)
    print(f"escrito en {destino.with_suffix('')}")


if __name__ == "__main__":
    main()
