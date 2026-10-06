"""
Resume una corrida: puntajes por método, por año y por zona.

No basta el promedio. Lo que distingue una mejora sistemática de una que cargan
dos zonas o un año bueno es la distribución, y por eso aquí se reporta por año y
por zona además del total.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate import evaluar_archivo  # noqa: E402
from experiments.protocol import COMMON_LEVELS, RESULTS  # noqa: E402

## La rejilla común va de 0.1 a 0.9, así que el intervalo que existe es el de 80%:
## pedir el de 90% exigiría los cuantiles 0.05 y 0.95, que no todos los métodos
## emiten de fábrica. Se piden los dos y se usan los que haya.
COLUMNAS = ["mean_pinball", "crps", "reliability", "cover80", "sharp80",
            "winkler80", "cover90", "sharp90", "winkler90", "mae", "pinball95",
            "seconds"]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("uso: report_run.py <corrida> [--completo]")
    ruta = Path(sys.argv[1])

    ## la rejilla común sirve para comparar métodos que emiten distinta cantidad
    ## de cuantiles; entre variantes del mismo método conviene la completa, que
    ## incluye las colas donde vive el argumento del artículo
    completo = "--completo" in sys.argv
    puntajes = evaluar_archivo(ruta, None if completo else list(COMMON_LEVELS))
    puntajes["anio"] = pd.to_datetime(puntajes.origin).dt.year
    columnas = [c for c in COLUMNAS if c in puntajes.columns]

    print(f"corrida: {ruta.name}")
    print(f"{puntajes.zone.nunique()} zonas · {puntajes.origin.nunique()} orígenes "
          f"· {len(puntajes):,d} pronósticos evaluados")
    print(f"niveles: {'rejilla completa' if completo else 'rejilla común'}\n")

    print("por método")
    print(puntajes.groupby("method")[columnas].mean().round(3).to_string())

    print("\npor año, pérdida pinball media")
    print(puntajes.pivot_table(index="anio", columns="method",
                               values="mean_pinball").round(2).to_string())

    cobertura = "cover90" if "cover90" in puntajes.columns else "cover80"
    print(f"\npor año, cobertura del intervalo de "
          f"{'90' if cobertura == 'cover90' else '80'}%")
    print(puntajes.pivot_table(index="anio", columns="method",
                               values=cobertura).round(3).to_string())

    metodos = sorted(puntajes.method.unique())
    if len(metodos) == 2:
        a, b = metodos
        por_zona = puntajes.pivot_table(index="zone", columns="method",
                                        values="mean_pinball")
        gana = (por_zona[a] < por_zona[b]).sum()
        print(f"\n{a} gana a {b} en {gana} de {len(por_zona)} zonas")

    salida = RESULTS / f"{ruta.stem}_scores_common.csv"
    puntajes.to_csv(salida, index=False)
    resumen = puntajes.groupby(["method", "anio"])[columnas].mean()
    resumen.round(6).to_csv(RESULTS / f"{ruta.stem}_por_anio.csv")
    print(f"\nescrito {salida.name} y {ruta.stem}_por_anio.csv")


if __name__ == "__main__":
    main()
