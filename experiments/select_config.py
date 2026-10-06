"""
Elige la configuración del método y la congela.

La regla se fija antes de mirar los resultados, y es deliberadamente simple:

  1. un solo puntaje propio, la pérdida pinball media sobre la rejilla común;
  2. entre las celdas cuyo puntaje cae dentro de un error estándar del mínimo, se
     toma la más simple —el gamma menor, luego la ventana más corta, luego el k
     menor—.

Lo que NO se hace: usar la cobertura como criterio de elección, porque un intervalo
siempre puede ganar cobertura ensanchándose; ni usar valores p como mecanismo de
ajuste. La cobertura se reporta como diagnóstico, al lado, y no decide nada.

La salida es results/selected_config.json, que el experimento lee y sin el cual
falla.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.protocol import (RESULTS, SELECTED_FILE,  # noqa: E402
                                  SELECTION_END, SELECTION_START)

PUNTAJE = "mean_pinball"
PARAMETROS = ["gamma", "window", "k", "separation"]


def regla_de_un_error_estandar(crudo: pd.DataFrame, columnas: list) -> dict:
    """
    Devuelve la celda elegida y el conjunto que queda dentro de un error estándar.

    El error estándar se calcula sobre la media por origen, no sobre las filas
    sueltas: las zonas del mismo día comparten choques y tratarlas como
    independientes encogería el error hasta volverlo decorativo.
    """
    por_origen = crudo.groupby(columnas + ["origen"])[PUNTAJE].mean()
    resumen = por_origen.groupby(columnas).agg(["mean", "sem", "count"])
    resumen.columns = ["puntaje", "error_estandar", "origenes"]

    mejor = resumen.puntaje.idxmin()
    umbral = (resumen.loc[mejor, "puntaje"]
              + resumen.loc[mejor, "error_estandar"])
    dentro = resumen[resumen.puntaje <= umbral].reset_index()

    ## entre las que no se distinguen del mínimo, la más simple
    orden = [c for c in ["gamma", "window", "k"] if c in columnas]
    elegida = dentro.sort_values(orden).iloc[0]
    return {"resumen": resumen, "dentro": dentro, "elegida": elegida,
            "mejor": mejor, "umbral": float(umbral)}


def main() -> None:
    crudo = pd.read_csv(RESULTS / "gamma_sweep_raw.csv", parse_dates=["origen"])
    columnas = [c for c in PARAMETROS if c in crudo.columns]
    if not columnas:
        raise SystemExit("el barrido no trae ninguna columna de parámetros")

    fallo = regla_de_un_error_estandar(crudo, columnas)
    resumen, dentro, elegida = fallo["resumen"], fallo["dentro"], fallo["elegida"]

    print(f"tramo de selección {SELECTION_START} a {SELECTION_END}")
    print(f"celdas evaluadas: {len(resumen)}\n")
    print(resumen.round(4).to_string())
    print(f"\nmejor puntaje: {fallo['mejor']} "
          f"({resumen.puntaje.min():.4f}), umbral de un error estándar: "
          f"{fallo['umbral']:.4f}")
    print(f"celdas dentro del umbral: {len(dentro)}")
    print(f"elegida, por ser la más simple del conjunto: "
          f"{ {c: elegida[c] for c in columnas} }")

    diagnostico = crudo.groupby(columnas)[
        ["reliability", "cover90", "sharp90", "pinball95"]].mean()
    print("\ndiagnóstico de la elegida, que NO entró en la regla:")
    clave = tuple(elegida[c] for c in columnas)
    print(diagnostico.loc[clave].round(4).to_string())

    SELECTED_FILE.parent.mkdir(exist_ok=True)
    with open(SELECTED_FILE, "w") as handle:
        json.dump({
            **{c: float(elegida[c]) if c != "k" else int(elegida[c])
               for c in columnas},
            "criterio": PUNTAJE,
            "regla": "un error estándar sobre la media por origen, desempate "
                     "por la celda más simple",
            "puntaje_elegida": float(elegida["puntaje"]),
            "puntaje_mejor": float(resumen.puntaje.min()),
            "umbral": fallo["umbral"],
            "celdas_dentro_del_umbral": int(len(dentro)),
            "tramo": [SELECTION_START, SELECTION_END],
            "origenes": int(resumen.origenes.max()),
        }, handle, indent=2, ensure_ascii=False)
    resumen.round(6).to_csv(RESULTS / "selection_summary.csv")
    print(f"\ncongelado en {SELECTED_FILE.name}")


if __name__ == "__main__":
    main()
