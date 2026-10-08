"""
Le da escenarios a cualquier método del banco, con la misma receta para todos.

Existe porque excluir a los modelos de fundación de la pregunta del costo del día fue
trato desigual, no una limitación de esos modelos.

El argumento que los excluía era: entregan bordes por hora, y los cuantiles no se suman,
así que no pueden formar una distribución del costo del día. La primera mitad es cierta.
La conclusión no se sigue, y la prueba es que a los comparativos ingenuos y a LEAR —que
tampoco entregan escenarios, porque son pronósticos puntuales— sí se les armaron
escenarios, con esta receta:

    trayectoria_i = mediana de hoy + vector de error del día pasado i

El error de un día real tiene forma real —chico de madrugada, grande a las siete de la
tarde— así que las trayectorias salen coherentes sin inventar nada. Es la construcción
estándar de un ensamble a partir de un pronóstico puntual.

Dársela a unos métodos y negársela a otros inclina el cuadro. Aquí se le da a TODOS,
incluido el análogo sobre su propia mediana, que además permite separar dos cosas que
estaban confundidas:

- cuánto vale tener escenarios de cualquier clase, y
- cuánto vale además que los del análogo vengan de continuaciones reales y no de errores
  sumados a un punto.

Nada mira más allá del origen: los errores que entran son de días ya observados.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.forecast_store import ForecastStore  # noqa: E402
from experiments.protocol import RESULTS  # noqa: E402
from experiments.scores import LEVELS  # noqa: E402

VENTANA_ERRORES = 60


def de_una_zona(pronosticos: pd.DataFrame, almacen: ForecastStore,
                ventana: int = VENTANA_ERRORES) -> int:
    """Recorre los orígenes en orden y emite el centro más los errores ya vistos."""
    escritos = 0
    for metodo, bloque in pronosticos.groupby("method", sort=False):
        errores: list = []
        for origen, p in sorted(bloque.groupby("origin"), key=lambda kv: kv[0]):
            p = p.sort_values("step")
            observado = p["observed"].to_numpy()
            mediana = p["q0.5"].to_numpy()
            if np.isnan(mediana).any():
                continue

            if errores:
                miembros = mediana[None, :] + np.array(errores[-ventana:])
                cuantiles = np.quantile(miembros, LEVELS, axis=0)
                almacen.add(p.zone.iloc[0], origen, f"{metodo}+errores",
                            cuantiles, observado, miembros, float("nan"))
                escritos += 1

            errores.append(observado - mediana)
    return escritos


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corrida")
    p.add_argument("--ventana", type=int, default=VENTANA_ERRORES)
    p.add_argument("--salida", default=None)
    args = p.parse_args()

    ruta = Path(args.corrida)
    partes = sorted((ruta if ruta.is_dir() else ruta.with_suffix("")).glob("*.parquet"))
    destino = Path(args.salida or RESULTS / f"{ruta.name}_errores.parquet")
    almacen = ForecastStore(LEVELS, destino)
    for borrado in almacen.limpiar():
        print(f"  (se borró {Path(borrado).name})")

    print(f"{len(partes)} zonas, ventana de {args.ventana} días de error", flush=True)
    for numero, parte in enumerate(partes, 1):
        escritos = de_una_zona(pd.read_parquet(parte), almacen, args.ventana)
        almacen.flush(parte.stem)
        print(f"  {numero}/{len(partes)} {parte.stem:20s} {escritos:,d} pronósticos",
              flush=True)
    print(f"\nescrito en {destino.with_suffix('')}")


if __name__ == "__main__":
    main()
