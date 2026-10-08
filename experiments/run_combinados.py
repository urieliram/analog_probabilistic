"""
Dos maneras de aprovechar al análogo junto al mejor modelo de fundación.

Diagnóstico que las motiva: el análogo pierde sobre todo en el centro (error absoluto
medio de 210 pesos contra 189 de t0-beta), no en el abanico.

``Combinado-t0``  60 escenarios de Analog-mezcla, sorteados con semilla fija por día, más
                  los 60 de t0-beta+errores, en un solo conjunto con el mismo peso. Pregunta:
                  ¿aporta el análogo algo al mejor del banco? En pronóstico de precios
                  combinar suele ganarle al mejor componente cuando se equivocan en días
                  distintos.
``Hibrido-t0``    el centro de t0-beta+errores con el abanico del análogo:
                  mediana_t0 + (escenario_análogo - mediana_análogo), 160 escenarios.
                  Pregunta: ¿describen los episodios reales del análogo la incertidumbre
                  mejor que los errores recientes de t0-beta, a igual centro? Es la tesis
                  original del trabajo, probada sin el defecto del centro.

Hipótesis declaradas antes de correr, con medida principal la CRPS ponderada de la
cuenta del día (pesos uniforme y cola alta), todos con la calibración por nivel y bloque:
H5. Combinado-t0 es mejor que t0-beta+errores.
H6. Hibrido-t0 es mejor que t0-beta+errores.
Se reportan las dos, ganen o pierdan.
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

ANALOGO = ("picos_prueba", "Analog-mezcla")
RIVAL = ("err_t0-beta", "t0-beta+errores")
TAMANO = 60


def _bloques(corrida, metodo, zona):
    m = pd.read_parquet(RESULTS / f"{corrida}_members" / f"{zona}.parquet")
    m = m[m.method == metodo]
    cols = member_columns(m)
    return {o: member_matrix(b.sort_values("step"), cols) for o, b in m.groupby("origin")}


def main():
    destino = RESULTS / "combinados.parquet"
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
            semilla = zlib.crc32(f"{zona}|{origen}".encode())
            elegidos = np.random.default_rng(semilla).choice(len(A), min(TAMANO, len(A)),
                                                             replace=False)
            combinado = np.vstack([A[elegidos], B])
            hibrido = np.median(B, axis=0) + (A - np.median(A, axis=0))
            for nombre, miembros in [("Combinado-t0", combinado), ("Hibrido-t0", hibrido)]:
                almacen.add(zona, origen, nombre, ensemble_quantiles(miembros, LEVELS),
                            y, miembros, 0.0)
        almacen.flush(zona)
        print(f"  {n}/{len(zonas)} {zona}", flush=True)
    print(f"escrito en {destino.with_suffix('')}")


if __name__ == "__main__":
    main()
