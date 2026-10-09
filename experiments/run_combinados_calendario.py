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


def analiza():
    """H12 y H13 con los puntajes por día ya calibrados, y un diagnóstico del centro."""
    from experiments.calibra_escenarios import DETALLE, NIVELES
    from experiments.curva_tau import PESOS, compara
    partes = [("_calendario", ["Analog-mezcla-cal7"]),
              ("_err_cal", ["t0-beta+errores-cal7", "PatchTST-FM+errores-cal7"]),
              ("_combinados_cal", ["Combinado-cal", "Centro-promedio-cal"])]
    t = pd.concat(pd.read_parquet(str(DETALLE).replace(".parquet", f"{sufijo}.parquet"))
                  .query("metodo in @metodos") for sufijo, metodos in partes)
    t = t[t.variante == "nivel_bloque"].copy()
    p = t[[f"pin_{a:.2f}" for a in NIVELES]].to_numpy()
    for nombre, peso in PESOS.items():
        t[f"qw_{nombre}"] = 2 * (p * peso(NIVELES)).mean(axis=1)
    t["cubre"] = (t.costo_real >= t["cq_0.05"]) & (t.costo_real <= t["cq_0.95"])
    print("costo del día, calibración por nivel y bloque (menos es mejor):")
    print(t.groupby("metodo").agg(
        zona_dias=("origen", "size"), cobertura=("cubre", "mean"),
        toda=("qw_uniforme", "mean"), cola_alta=("qw_cola_alta", "mean"),
        cola_baja=("qw_cola_baja", "mean")).round(3).to_string())
    filas = []
    for combo in ("Combinado-cal", "Centro-promedio-cal"):
        for rival in ("t0-beta+errores-cal7", "Analog-mezcla-cal7",
                      "PatchTST-FM+errores-cal7"):
            for medida in ("qw_uniforme", "qw_cola_alta", "qw_cola_baja"):
                filas.append(compara(t, combo, rival, medida))
    r = pd.DataFrame(filas)
    print("\ncombinación menos rival, en por ciento del rival (estadístico); negativo: "
          "gana la combinación")
    for (combo, rival), g in r.groupby(["a", "b"], sort=False):
        print(f"   {combo:20s} vs {rival:26s} " + "  ".join(
            f"{x.medida[3:]} {x.pct:+.1f} ({x.t:+.1f})" for x in g.itertuples()))
    print("\nzona por zona, cola alta, contra t0-beta+errores-cal7:")
    for combo in ("Combinado-cal", "Centro-promedio-cal"):
        z = [compara(t[t.zona == zona], combo, "t0-beta+errores-cal7", "qw_cola_alta")
             for zona in sorted(t.zona.unique())]
        print(f"   {combo:20s} gana en {sum(x['pct'] < 0 for x in z)} de {len(z)}; "
              f"con estadístico menor que -1.96 en {sum(x['t'] < -1.96 for x in z)}, "
              f"mayor que 1.96 en {sum(x['t'] > 1.96 for x in z)}")
    r.round(4).to_csv(RESULTS / "publicacion" / "cuadro_combinados_calendario.csv",
                      index=False)
    ## diagnóstico, no declarado: error absoluto medio del centro promediado
    def medianas(corrida, metodo):
        partes = []
        for f in sorted((RESULTS / corrida).glob("*.parquet")):
            m = pd.read_parquet(f, columns=["zone", "origin", "method", "step", "q0.5",
                                            "observed"])
            partes.append(m[m.method == metodo])
        return pd.concat(partes).set_index(["zone", "origin", "step"])
    a = medianas("calendario", "Analog-mezcla-cal7")
    b = medianas("fundacion_t0-beta", "t0-beta")
    comun = a.index.intersection(b.index)
    ya, yb, y = a.loc[comun]["q0.5"], b.loc[comun]["q0.5"], b.loc[comun]["observed"]
    print(f"\ndiagnóstico del centro, {len(comun):,d} horas: análogo con calendario "
          f"{(ya - y).abs().mean():.1f}, t0-beta {(yb - y).abs().mean():.1f}, promedio de "
          f"los dos {((ya + yb) / 2 - y).abs().mean():.1f}; correlación de sus errores "
          f"{np.corrcoef(ya - y, yb - y)[0, 1]:.2f}")


if __name__ == "__main__":
    analiza() if sys.argv[1:] == ["analiza"] else main()
