"""
Califica un archivo de pronósticos. Es el único lugar donde se calculan puntajes.

Dos comparaciones, y la diferencia entre ellas importa:

  en crudo      sobre los nueve niveles que todos los métodos emiten de fábrica,
                de 0.1 a 0.9. Un modelo que entrega noventa y nueve cuantiles y
                otro que entrega nueve no se comparan calculando a cada uno con su
                propia resolución: eso mezcla la calidad del modelo con la finura
                del puntaje.

  calibrada     sobre la rejilla completa, con las colas, porque la capa de
                calibración deja a todos en los mismos niveles.

Escribe una fila por zona, origen y método.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.forecast_store import (load_forecasts,  # noqa: E402
                                        load_members, member_columns,
                                        quantile_columns)
from experiments.protocol import COMMON_LEVELS, RESULTS  # noqa: E402
from experiments.scores import evaluate  # noqa: E402


def evaluar_archivo(path: Path, levels=None) -> pd.DataFrame:
    """
    Puntajes por zona, origen y método, sobre los niveles pedidos.

    Se ordena una vez y se trabaja con matrices. La versión ingenua filtraba los
    miembros con un barrido completo de la tabla por cada pronóstico: con noventa
    mil pronósticos sobre dos millones de filas, eso es cuadrático y no termina.
    """
    pronosticos = load_forecasts(path)
    columnas, disponibles = quantile_columns(pronosticos)
    niveles = [a for a in (levels or disponibles)
               if round(a, 4) in [round(d, 4) for d in disponibles]]
    if not niveles:
        raise ValueError("ninguno de los niveles pedidos está en el archivo")
    columnas_usadas = [f"q{a:g}" for a in niveles]

    llave = ["zone", "origin", "method"]
    pronosticos = pronosticos.sort_values(llave + ["step"], kind="stable")
    claves = pronosticos[llave].drop_duplicates().reset_index(drop=True)
    pasos = len(pronosticos) // len(claves)
    if len(pronosticos) != pasos * len(claves):
        raise ValueError("hay pronósticos con distinto número de pasos")

    cuantiles = (pronosticos[columnas_usadas].to_numpy()
                 .reshape(len(claves), pasos, len(niveles)))
    observado = pronosticos["observed"].to_numpy().reshape(len(claves), pasos)
    segundos = pronosticos["seconds"].to_numpy().reshape(len(claves), pasos)[:, 0]

    muestras = None
    miembros = load_members(path)
    if miembros is not None:
        columnas_miembro = member_columns(miembros)
        miembros = miembros.sort_values(llave + ["step"], kind="stable")
        suyas = miembros[llave].drop_duplicates().reset_index(drop=True)
        if suyas.equals(claves) and len(miembros) == pasos * len(claves):
            muestras = (miembros[columnas_miembro].to_numpy()
                        .reshape(len(claves), pasos, len(columnas_miembro)))

    filas = []
    for fila in range(len(claves)):
        muestra = muestras[fila].T if muestras is not None else None
        filas.append({
            "zone": claves.zone[fila], "origin": claves.origin[fila],
            "method": claves.method[fila], "seconds": float(segundos[fila]),
            **evaluate(observado[fila], cuantiles[fila].T, niveles, muestra),
        })
    return pd.DataFrame(filas)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("uso: evaluate.py <archivo de pronósticos> [--completo]")
    ruta = Path(sys.argv[1])
    completo = "--completo" in sys.argv

    niveles = None if completo else list(COMMON_LEVELS)
    puntajes = evaluar_archivo(ruta, niveles)

    sufijo = "full" if completo else "common"
    salida = RESULTS / f"{ruta.stem}_scores_{sufijo}.csv"
    puntajes.to_csv(salida, index=False)

    resumen = puntajes.groupby("method")[
        [c for c in ["mean_pinball", "crps", "wis", "reliability", "cover90",
                     "sharp90", "winkler90", "mae", "seconds"]
         if c in puntajes.columns]].mean()
    print(f"niveles usados: {'rejilla completa' if completo else 'rejilla común'}")
    print(resumen.round(4).to_string())
    print(f"\nescrito {salida}")


if __name__ == "__main__":
    main()
