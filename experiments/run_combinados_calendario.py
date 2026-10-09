"""
El análogo con calendario junto al mejor modelo de fundación con calendario.

Con la misma información de calendario para todos, Analog-mezcla-cal7 empata con
t0-beta+errores-cal7 en el costo del día (+0.3 por ciento en la cola alta, estadístico
0.2). La literatura de combinación de pronósticos dice que combinar dos métodos de
exactitud parecida que se equivocan en días distintos suele ganarle a cada uno. La
combinación que ya está en el artículo (Combinado-t0) se probó sin calendario, cuando el
análogo iba 7 por ciento atrás, y empató con t0-beta.

Las dos combinaciones usan el mismo número de escenarios de cada método: 60, o menos al
principio del tramo, cuando t0-beta con calendario todavía tiene pocos errores pasados de
ese tipo de día. Los del análogo se sortean con semilla fija por zona y día.

``Combinado-cal``        los escenarios de los dos en un solo conjunto, con el mismo peso:
                         la receta de Combinado-t0, ahora con calendario.
``Centro-promedio-cal``  el centro es el promedio de las dos medianas, y alrededor van las
                         desviaciones de esos mismos escenarios respecto de la mediana de
                         su propio método. Combina los centros, que es donde estaba la
                         diferencia entre métodos, y junta los abanicos.

Hipótesis declaradas antes de correr, con medida principal la CRPS ponderada del costo
del día (pesos uniforme y cola alta), todos con la calibración por nivel y bloque de horas
y la prueba de Diebold y Mariano sobre los días:
H12. Combinado-cal es mejor que t0-beta+errores-cal7.
H13. Centro-promedio-cal es mejor que t0-beta+errores-cal7.
Se reportan las dos, ganen o pierdan, junto con la comparación contra Analog-mezcla-cal7.
"""

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.forecast_store import (ForecastStore, member_columns,  # noqa: E402
                                        member_matrix)
from analog_probabilistic.analog_probabilistic import ensemble_quantiles  # noqa: E402
from experiments.protocol import RESULTS  # noqa: E402
from experiments.scores import LEVELS  # noqa: E402

ANALOGO = ("calendario", "Analog-mezcla-cal7")
RIVAL = ("err_cal", "t0-beta+errores-cal7")
TAMANO = 60


def _bloques(corrida, metodo, zona):
    m = pd.read_parquet(RESULTS / f"{corrida}_members" / f"{zona}.parquet")
    m = m[m.method == metodo]
    cols = member_columns(m)
    return {o: member_matrix(b.sort_values("step"), cols) for o, b in m.groupby("origin")}


def combina(A, B, semilla):
    """Las dos combinaciones de un día, con el mismo número de escenarios de cada método."""
    rng = np.random.default_rng(semilla)
    n = min(TAMANO, len(A), len(B))
    a = A[rng.choice(len(A), n, replace=False)]
    b = B[rng.choice(len(B), n, replace=False)] if len(B) > n else B
    mediana_a, mediana_b = np.median(A, axis=0), np.median(B, axis=0)
    centro = (mediana_a + mediana_b) / 2
    juntos = np.vstack([a, b])
    promedio = centro + np.vstack([a - mediana_a, b - mediana_b])
    return juntos, promedio


def main():
    destino = RESULTS / "combinados_cal.parquet"
    almacen = ForecastStore(LEVELS, destino)
    zonas = sorted(p.stem for p in (RESULTS / f"{RIVAL[0]}_members").glob("*.parquet"))
    for n, zona in enumerate(zonas, 1):
        a = _bloques(*ANALOGO, zona)
        r = _bloques(*RIVAL, zona)
        obs = pd.read_parquet(RESULTS / RIVAL[0] / f"{zona}.parquet",
                              columns=["origin", "step", "observed"]).drop_duplicates(
            ["origin", "step"])
        observados = {o: b.sort_values("step")["observed"].to_numpy()
                      for o, b in obs.groupby("origin")}
        for origen in sorted(set(a) & set(r) & set(observados)):
            A, B, y = a[origen], r[origen], observados[origen]
            if A.shape[1] != 24 or B.shape[1] != 24 or len(B) < 2 or len(y) != 24:
                continue
            juntos, promedio = combina(A, B, zlib.crc32(f"{zona}|{origen}".encode()))
            for nombre, miembros in [("Combinado-cal", juntos),
                                     ("Centro-promedio-cal", promedio)]:
                almacen.add(zona, origen, nombre, ensemble_quantiles(miembros, LEVELS),
                            y, miembros, 0.0)
        almacen.flush(zona)
        print(f"  {n}/{len(zonas)} {zona}", flush=True)
    print(f"escrito en {destino.with_suffix('')}")


if __name__ == "__main__":
    main()
