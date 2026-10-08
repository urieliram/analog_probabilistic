"""
Mejorar el centro del análogo sin tocar su abanico: corrección de sesgo por nivel, e
híbridos con LEAR.

El diagnóstico (2026-10-08): el análogo pierde en el centro, no en el abanico. Sobrestima
los precios bajos (sesgo de +75 pesos en el decil más barato, contra +55 de t0-beta) y
el promedio de su centro con el de LEAR da un error absoluto medio de 197.7 contra 205.4
de LEAR y 209.8 del análogo, con errores correlacionados a 0.88.

Cuatro construcciones sobre los escenarios ya guardados de Analog-mezcla (A, 160) y los
de LEAR (L, 60, su centro más sus errores pasados):

``Analog-mezcla-sesgo``         A + c_h: a cada hora se le suma el error mediano
                                (observado menos mediana) que el análogo tuvo en los
                                últimos cien días en las horas del mismo tramo de nivel
                                pronosticado (diez tramos). Principio de retroalimentación
                                con calidad local: corrige el sesgo donde el método lo
                                tiene, no en promedio.
``Analog-mezcla-sesgo-bloque``  lo mismo, con cinco tramos de nivel dentro de cada bloque
                                de seis horas, como la calibración por nivel y bloque.
``Hibrido-LEAR``                mediana(L) + (A - mediana(A)): centro de LEAR, abanico del
                                análogo. Es el espejo de Hibrido-t0 con métodos propios.
``Hibrido-prom``                (mediana(A) + mediana(L)) / 2 + (A - mediana(A)): el
                                centro promedio de los dos métodos propios con el abanico
                                del análogo.

Hipótesis declaradas antes de correr, con el error absoluto medio del centro y la CRPS
ponderada de la cuenta del día calibrada por nivel y bloque:
H9.  La corrección de sesgo baja el error del centro 1 a 2% y quita la sobrestimación
     en los deciles baratos; la cuenta del día mejora un poco o empata.
H10. Hibrido-prom tiene un error de centro cercano a 198 y una cuenta del día mejor que
     LEAR y que Analog-mezcla; Hibrido-LEAR empata con LEAR.
Ninguna corrección usa información posterior al origen.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_probabilistic.analog_probabilistic import ensemble_quantiles  # noqa: E402
from experiments.forecast_store import (ForecastStore, member_columns,  # noqa: E402
                                        member_matrix)
from experiments.protocol import RESULTS  # noqa: E402
from experiments.scores import LEVELS  # noqa: E402

ANALOGO = ("picos_prueba", "Analog-mezcla")
LEAR = ("lear_prueba", "LEAR")
VENTANA = 100
HISTORIA_MINIMA = 30
TRAMOS = 10
TRAMOS_BLOQUE = 5
BLOQUES = (range(0, 6), range(6, 12), range(12, 18), range(18, 24))


def _bloques(corrida, metodo, zona):
    m = pd.read_parquet(RESULTS / f"{corrida}_members" / f"{zona}.parquet")
    m = m[m.method == metodo]
    cols = member_columns(m)
    return {o: member_matrix(b.sort_values("step"), cols) for o, b in m.groupby("origin")}


def _correccion(med_pasado, err_pasado, med_hoy, tramos):
    """Error mediano por tramo de nivel de la mediana, aplicado a las horas de hoy."""
    cortes = np.quantile(med_pasado, np.linspace(0, 1, tramos + 1)[1:-1])
    tramo_pasado = np.searchsorted(cortes, med_pasado)
    tramo_hoy = np.searchsorted(cortes, med_hoy)
    c = np.zeros(len(med_hoy))
    for t in np.unique(tramo_hoy):
        dentro = tramo_pasado == t
        if dentro.sum() >= 10:
            c[tramo_hoy == t] = np.median(err_pasado[dentro])
    return c


def main():
    destino = RESULTS / "centros.parquet"
    almacen = ForecastStore(LEVELS, destino)
    zonas = sorted(p.stem for p in (RESULTS / f"{LEAR[0]}_members").glob("*.parquet"))
    for n, zona in enumerate(zonas, 1):
        A = _bloques(*ANALOGO, zona)
        L = _bloques(*LEAR, zona)
        obs = pd.read_parquet(RESULTS / ANALOGO[0] / f"{zona}.parquet",
                              columns=["origin", "step", "observed"]).drop_duplicates(
            ["origin", "step"])
        observados = {o: b.sort_values("step")["observed"].to_numpy()
                      for o, b in obs.groupby("origin")}
        H = {"med": [], "obs": []}
        for origen in sorted(set(A) & set(observados)):
            a, y = A[origen], observados[origen]
            if a.shape[1] != 24 or len(y) != 24:
                continue
            med_a = np.median(a, axis=0)
            desv = a - med_a
            salida = {}
            if len(H["med"]) >= HISTORIA_MINIMA:
                Hm = np.array(H["med"][-VENTANA:])
                He = np.array(H["obs"][-VENTANA:]) - Hm
                c = _correccion(Hm.ravel(), He.ravel(), med_a, TRAMOS)
                salida["Analog-mezcla-sesgo"] = a + c
                cb = np.zeros(24)
                for bloque in BLOQUES:
                    h = list(bloque)
                    cb[h] = _correccion(Hm[:, h].ravel(), He[:, h].ravel(), med_a[h],
                                        TRAMOS_BLOQUE)
                salida["Analog-mezcla-sesgo-bloque"] = a + cb
            if origen in L and L[origen].shape[1] == 24 and len(L[origen]) >= 2:
                med_l = np.median(L[origen], axis=0)
                salida["Hibrido-LEAR"] = med_l + desv
                salida["Hibrido-prom"] = (med_a + med_l) / 2 + desv
            for nombre, miembros in salida.items():
                almacen.add(zona, origen, nombre, ensemble_quantiles(miembros, LEVELS),
                            y, miembros, 0.0)
            H["med"].append(med_a)
            H["obs"].append(y)
        almacen.flush(zona)
        print(f"  {n}/{len(zonas)} {zona}", flush=True)
    print(f"escrito en {destino.with_suffix('')}")


if __name__ == "__main__":
    main()
