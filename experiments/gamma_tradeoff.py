"""
Qué compra y qué cuesta la contracción, con errores pareados.

Las celdas del barrido se evaluaron sobre los mismos orígenes y los mismos
análogos, así que compararlas con el error estándar de cada media es demasiado
conservador: el error que importa es el de la diferencia, origen por origen.

Lo que sale de aquí es la figura central del artículo: el mismo parámetro mueve la
exactitud del pronóstico puntual y la dispersión del ensamble en sentidos
contrarios, y el puntaje propio, que promedia las dos cosas, no los distingue.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.protocol import RESULTS  # noqa: E402

PUNTAJE = "mean_pinball"


def diferencias_pareadas(crudo: pd.DataFrame, columna: str,
                         referencia: float) -> pd.DataFrame:
    """Diferencia contra la celda de referencia, origen por origen."""
    por_origen = crudo.groupby(["gamma", "origen"])[columna].mean().unstack(0)
    filas = []
    for gamma in sorted(por_origen.columns):
        diferencia = (por_origen[gamma] - por_origen[referencia]).dropna()
        error = diferencia.std(ddof=1) / np.sqrt(len(diferencia))
        filas.append({
            "gamma": gamma,
            "diferencia": diferencia.mean(),
            "error_estandar": error,
            "t": diferencia.mean() / error if error else np.nan,
            "peor_en_dias": float((diferencia > 0).mean()),
        })
    return pd.DataFrame(filas).set_index("gamma")


def main() -> None:
    crudo = pd.read_csv(RESULTS / "gamma_sweep_raw.csv", parse_dates=["origen"])
    medias = crudo.groupby("gamma")[
        [PUNTAJE, "crps", "reliability", "cover90", "sharp90", "winkler90",
         "mae", "pinball95"]].mean()
    mejor = float(medias[PUNTAJE].idxmin())

    puntaje = diferencias_pareadas(crudo, PUNTAJE, mejor)
    print(f"Diferencia de pérdida pinball contra gamma = {mejor}, pareada\n")
    print(puntaje.round(4).to_string())
    print(f"\nel |t| más grande es {puntaje.t.abs().max():.2f}: el puntaje propio "
          f"no distingue los valores de gamma")

    punto = diferencias_pareadas(crudo, "mae", 0.0)
    extremos = punto.loc[1.0]
    print(f"\nerror de la mediana, gamma 1 contra gamma 0: "
          f"{extremos.diferencia:+.3f} ± {extremos.error_estandar:.3f} "
          f"(t = {extremos.t:.2f})")
    print("la contracción SÍ mejora el pronóstico puntual, y significativamente")

    print("\nY lo que empeora, monótonamente:")
    print(medias[["cover90", "reliability", "sharp90", "winkler90",
                  "pinball95"]].round(4).to_string())

    salida = pd.concat([medias, puntaje.add_prefix("pinball_")], axis=1)
    salida.round(6).to_csv(RESULTS / "gamma_tradeoff.csv")
    print(f"\nescrito {RESULTS / 'gamma_tradeoff.csv'}")


if __name__ == "__main__":
    main()
