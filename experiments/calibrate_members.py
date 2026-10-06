"""
Calibra el ensamble sin romper las trayectorias.

Recalibrar los cuantiles uno por uno arregla la cobertura hora por hora y deshace
la estructura de caminos: los cuantiles recalibrados ya no son un conjunto de
escenarios, y los puntajes de trayectoria dejan de significar algo. Aquí se escala
a los miembros alrededor de la mediana,

    Y~(i, tau) = m(tau) + c(tau) * (Y(i, tau) - m(tau))

de modo que el abanico se ensancha y el orden de los escenarios no cambia: el
miembro que iba más alto sigue yendo más alto.

El factor c se estima con los errores de orígenes YA observados y nada más: para
cada paso del horizonte, el cociente entre la dispersión que habría hecho falta y
la que el ensamble traía. Nada posterior al origen entra en el cálculo.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.forecast_store import (ForecastStore,  # noqa: E402
                                        load_forecasts, load_members,
                                        member_columns, quantile_columns)
from experiments.protocol import RESULTS  # noqa: E402
from experiments.scores import LEVELS  # noqa: E402

ORIGENES_PASADOS = 60
MIN_DISPERSION = 1e-6


def factor_por_paso(errores: np.ndarray, dispersiones: np.ndarray,
                    nivel: float = 0.9) -> np.ndarray:
    """
    Cuánto habría que ensanchar cada paso para cubrir lo que promete.

    Se compara el cuantil del error absoluto estandarizado con el que tendría una
    distribución bien calibrada: si el ensamble se queda corto, el factor sube.
    """
    estandarizado = np.abs(errores) / np.maximum(dispersiones, MIN_DISPERSION)
    return np.quantile(estandarizado, nivel, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archivo")
    parser.add_argument("--metodo", default="Analog")
    parser.add_argument("--ventana", type=int, default=ORIGENES_PASADOS)
    args = parser.parse_args()

    ruta = Path(args.archivo)
    pronosticos = load_forecasts(ruta)
    miembros = load_members(ruta)
    if miembros is None:
        raise SystemExit("ese archivo no trae miembros: no hay qué escalar")

    columnas_q, niveles = quantile_columns(pronosticos)
    columnas_m = member_columns(miembros)
    mediana = f"q{0.5:g}"

    destino = ruta.with_name(ruta.stem + "_calibrado.parquet")
    almacen = ForecastStore(niveles, destino)
    hechos = 0

    for zona, bloque in pronosticos[pronosticos.method == args.metodo].groupby(
            "zone", sort=False):
        bloque_m = miembros[(miembros.zone == zona)
                            & (miembros.method == args.metodo)]
        origenes = sorted(bloque.origin.unique())

        historial_error, historial_dispersion = [], []
        for origen in origenes:
            paso = bloque[bloque.origin == origen].sort_values("step")
            muestra = bloque_m[bloque_m.origin == origen].sort_values("step")
            if muestra.empty:
                continue

            ensamble = muestra[columnas_m].to_numpy().T
            centro = paso[mediana].to_numpy()
            observado = paso["observed"].to_numpy()
            dispersion = (np.percentile(ensamble, 90, axis=0)
                          - np.percentile(ensamble, 10, axis=0)) / 2

            if len(historial_error) >= args.ventana:
                c = factor_por_paso(
                    np.array(historial_error[-args.ventana:]),
                    np.array(historial_dispersion[-args.ventana:]))
                escalado = centro + c * (ensamble - centro)
                cuantiles = np.quantile(escalado, niveles, axis=0)
                almacen.add(zona, origen, f"{args.metodo}-calibrado", cuantiles,
                            observado, escalado,
                            float(paso["seconds"].iloc[0]))
                hechos += 1

            historial_error.append(observado - centro)
            historial_dispersion.append(dispersion)

    if not hechos:
        raise SystemExit("no hubo orígenes suficientes para calibrar")
    print(almacen.save(), f"| pronósticos calibrados: {hechos}")


if __name__ == "__main__":
    main()
