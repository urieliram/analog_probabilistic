"""
Qué tan persistente es la diferencia diaria de pérdidas que entra a la prueba de
Diebold y Mariano, y qué le pasa al estadístico si la varianza usa más rezagos.

Coroneo e Iacone (2025) muestran que la prueba pierde potencia y rechaza en falso
cuando la diferencia de pérdidas es muy persistente, cerca de una raíz unitaria. Aquí
se mide, para las comparaciones principales del artículo, la correlación de la
diferencia diaria (sumada sobre las zonas) con la del día anterior y con la de hace
una semana, y el estadístico con 3, 7, 14, 30 y 60 rezagos de Newey y West. El
artículo usa 3.

    python experiments/persistencia_dm.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.calibra_escenarios import DETALLE, NIVELES  # noqa: E402
from experiments.curva_tau import PESOS  # noqa: E402
from experiments.protocol import RESULTS  # noqa: E402
from experiments.scores import diebold_mariano  # noqa: E402

PARTES = [("_calendario", ["Analog-mezcla-cal7"]),
          ("_err_cal", ["t0-beta+errores-cal7", "PatchTST-FM+errores-cal7"]),
          ("_combinados_cal", ["Centro-promedio-cal"]),
          ("", ["LEAR", "Analog-mezcla"])]
COMPARACIONES = [("Analog-mezcla-cal7", "LEAR"),
                 ("Analog-mezcla-cal7", "t0-beta+errores-cal7"),
                 ("Analog-mezcla-cal7", "PatchTST-FM+errores-cal7"),
                 ("Centro-promedio-cal", "t0-beta+errores-cal7"),
                 ("Analog-mezcla-cal7", "Analog-mezcla")]
REZAGOS = (3, 7, 14, 30, 60)


def _puntajes():
    cols = ["metodo", "variante", "zona", "origen"] + [f"pin_{a:.2f}" for a in NIVELES]
    t = pd.concat(
        pd.read_parquet(str(DETALLE).replace(".parquet", f"{s}.parquet"), columns=cols)
        .query("metodo in @m") for s, m in PARTES)
    t = t[t.variante == "nivel_bloque"].copy()
    p = t[[f"pin_{a:.2f}" for a in NIVELES]].to_numpy()
    for nombre in ("uniforme", "cola_alta"):
        t[f"qw_{nombre}"] = 2 * (p * PESOS[nombre](NIVELES)).mean(axis=1)
    return t


def _diferencia_diaria(t, a, b, col):
    pa = t[t.metodo == a].set_index(["zona", "origen"])[col]
    pb = t[t.metodo == b].set_index(["zona", "origen"])[col]
    comun = pa.index.intersection(pb.index)
    return (pa.loc[comun] - pb.loc[comun]).groupby(level="origen").sum().sort_index()


def _autocorrelacion(x, k):
    x = x - x.mean()
    return float((x[k:] * x[:-k]).sum() / (x * x).sum())


def main():
    t = _puntajes()
    filas = []
    for a, b in COMPARACIONES:
        for col in ("qw_cola_alta", "qw_uniforme"):
            d = _diferencia_diaria(t, a, b, col).to_numpy()
            fila = {"a": a, "b": b, "medida": col, "dias": len(d),
                    "rho_1": _autocorrelacion(d, 1), "rho_7": _autocorrelacion(d, 7)}
            for L in REZAGOS:
                fila[f"t_{L}"] = diebold_mariano(d, lags=L)[0]
            filas.append(fila)
    r = pd.DataFrame(filas)
    print(r.round(2).to_string(index=False))
    r.round(4).to_csv(RESULTS / "publicacion" / "cuadro_persistencia_dm.csv", index=False)


if __name__ == "__main__":
    main()
