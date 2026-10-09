"""
El pronóstico puntual: la mediana de cada método, por hora y en el costo del día.

Es donde brillan los modelos de fundación, y el artículo tiene que mostrarlo junto con
lo que pasa en lo que paga el comprador. Sobre las horas que tienen todos los métodos:
error absoluto medio y raíz del error cuadrático medio por hora, sesgo medio (pronóstico
menos real), fracción de horas en que el real queda por debajo de la mediana, y error y
sesgo del costo del día puntual, que es la suma de las 24 medianas comparada con el costo
real (comprar un megawatt-hora en cada hora). Además el rMAE de Lago et al. (2021): el
error absoluto medio de cada zona entre el del pronóstico ingenuo (la misma hora del día
anterior, o de la semana anterior para lunes, sábado y domingo), promediado entre zonas;
y el sMAPE, que se reporta para mostrar que pesa igual las horas baratas y las caras. El
MAPE no se calcula: hay horas con precio cero o negativo, y se cuentan. Las
comparaciones usan la prueba de Diebold y Mariano sobre las diferencias sumadas por día
de decisión.

Escribe ``cuadro_pronostico_puntual.csv`` y ``cuadro_pronostico_puntual_pruebas.csv``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.curva_tau import diebold_mariano  # noqa: E402
from experiments.protocol import RESULTS  # noqa: E402

FUENTES = {"Analog-mix con calendario": ("calendario", "Analog-mezcla-cal7"),
           "t0-beta": ("fundacion_t0-beta", "t0-beta"),
           "PatchTST-FM": ("fundacion_PatchTST-FM", "PatchTST-FM"),
           "LEAR": ("lear_prueba", "LEAR")}
PROMEDIO = "Promedio de t0-beta y el análogo"
PRUEBAS = [(PROMEDIO, "t0-beta"), ("Analog-mix con calendario", "t0-beta"),
           ("Analog-mix con calendario", "LEAR"), (PROMEDIO, "PatchTST-FM")]


def medianas(corrida, metodo):
    partes = []
    for f in sorted((RESULTS / corrida).glob("*.parquet")):
        t = pd.read_parquet(f, columns=["zone", "origin", "method", "step", "q0.5",
                                        "observed"])
        partes.append(t[t.method == metodo])
    return pd.concat(partes).set_index(["zone", "origin", "step"])


def datos():
    P = {k: medianas(*v) for k, v in FUENTES.items()}
    comun = None
    for v in P.values():
        comun = v.index if comun is None else comun.intersection(v.index)
    D = pd.DataFrame({k: v.loc[comun]["q0.5"].to_numpy() for k, v in P.items()},
                     index=comun)
    D["y"] = P["LEAR"].loc[comun]["observed"].to_numpy()
    D[PROMEDIO] = (D["t0-beta"] + D["Analog-mix con calendario"]) / 2
    return D


def main():
    D = datos()
    metodos = [k for k in D.columns if k != "y"]
    dia = D.groupby(level=["zone", "origin"]).sum()
    filas = []
    for k in metodos:
        e, ed = D[k] - D.y, dia[k] - dia.y
        filas.append({"metodo": k, "mae_hora": e.abs().mean(),
                      "rmse_hora": np.sqrt((e ** 2).mean()), "sesgo_hora": e.mean(),
                      "real_bajo_mediana": (D.y < D[k]).mean(),
                      "mae_costo_dia": ed.abs().mean(), "sesgo_costo_dia": ed.mean()})
    tabla = pd.DataFrame(filas)
    ## rMAE de Lago et al.: por zona, contra el ingenuo, sobre las horas que lo tienen
    t = D.reset_index()
    t["hora"] = pd.to_datetime(t.origin) + pd.to_timedelta(t.step, unit="h")
    y = t.set_index(["zone", "hora"]).y
    def rezago(h):
        r = y.copy()
        r.index = pd.MultiIndex.from_arrays([r.index.get_level_values(0),
                                             r.index.get_level_values(1)
                                             + pd.Timedelta(hours=h)])
        return r.reindex(pd.MultiIndex.from_arrays([t.zone, t.hora])).to_numpy()
    t["ingenuo"] = np.where(t.hora.dt.dayofweek.isin([0, 5, 6]), rezago(168), rezago(24))
    t = t.dropna(subset=["ingenuo"])
    escala = (t.ingenuo - t.y).abs().groupby(t.zone).mean()
    tabla["rmae_hora"] = [((t[k] - t.y).abs().groupby(t.zone).mean() / escala).mean()
                          for k in metodos]
    denominador = {k: D[k].abs() + D.y.abs() for k in metodos}
    tabla["smape_hora"] = [100 * np.where(denominador[k] > 0, 2 * (D[k] - D.y).abs()
                                          / denominador[k], 0).mean() for k in metodos]
    tabla["rmse_costo_dia"] = [np.sqrt(((dia[k] - dia.y) ** 2).mean()) for k in metodos]
    print(f"{len(D):,d} horas comunes; {len(t):,d} con ingenuo para el rMAE; precio cero "
          f"o negativo en {(D.y <= 0).sum():,d}, de valor absoluto menor que 1 en "
          f"{(D.y.abs() < 1).sum():,d}")
    print(tabla.round(3).to_string(index=False))
    pruebas = []
    for a, b in PRUEBAS:
        for medida, fa, fb in [
                ("error absoluto por hora", (D[a] - D.y).abs(), (D[b] - D.y).abs()),
                ("error cuadrático por hora", (D[a] - D.y) ** 2, (D[b] - D.y) ** 2),
                ("error absoluto del costo del día", (dia[a] - dia.y).abs(),
                 (dia[b] - dia.y).abs()),
                ("error cuadrático del costo del día", (dia[a] - dia.y) ** 2,
                 (dia[b] - dia.y) ** 2),
                ("sMAPE por hora",
                 pd.Series(np.where(D[a].abs() + D.y.abs() > 0, 2 * (D[a] - D.y).abs()
                                    / (D[a].abs() + D.y.abs()), 0), index=D.index),
                 pd.Series(np.where(D[b].abs() + D.y.abs() > 0, 2 * (D[b] - D.y).abs()
                                    / (D[b].abs() + D.y.abs()), 0), index=D.index))]:
            dif = (fa - fb).groupby(level="origin").sum().sort_index()
            base = fb.groupby(level="origin").sum().mean()
            t, _ = diebold_mariano(dif.to_numpy())
            pruebas.append({"a": a, "b": b, "medida": medida,
                            "pct": 100 * dif.mean() / base, "t": t})
    pruebas = pd.DataFrame(pruebas)
    print(pruebas.round(2).to_string(index=False))
    destino = RESULTS / "publicacion"
    tabla.round(4).to_csv(destino / "cuadro_pronostico_puntual.csv", index=False)
    pruebas.round(4).to_csv(destino / "cuadro_pronostico_puntual_pruebas.csv",
                            index=False)


if __name__ == "__main__":
    main()
