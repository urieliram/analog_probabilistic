"""
A qué hora se publican de verdad los precios del día en adelanto.

El Manual de Mercado de Energía de Corto Plazo fija el cierre de ofertas a las
10:00 del día anterior al de operación y la publicación de resultados a las 17:00.
Lo segundo se puede medir: cada archivo del CENACE lleva adentro su hora de
generación.

Esa marca prueba que el archivo no existía antes de esa hora, no que apareciera en
el portal en ese momento. Se usa como aproximación de la disponibilidad pública y
así se nombra en el artículo.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.protocol import RESULTS  # noqa: E402

ARCHIVO = Path(__file__).resolve().parents[1] / "data" / "publication_stamps.csv"
CIERRE_OFERTAS = 10.0


def main() -> None:
    sellos = pd.read_csv(ARCHIVO, parse_dates=["entrega", "generado"]).dropna()
    sellos["anticipacion_dias"] = (
        sellos.entrega.dt.normalize() - sellos.generado.dt.normalize()).dt.days
    sellos["hora"] = (sellos.generado.dt.hour
                      + sellos.generado.dt.minute / 60)
    sellos["anio"] = sellos.generado.dt.year

    ## sólo interesa la publicación de la víspera, que es la que el operador usa
    vispera = sellos[sellos.anticipacion_dias == 1]

    resumen = vispera.groupby("anio")["hora"].describe(
        percentiles=[0.5, 0.9, 0.99])[["count", "min", "50%", "90%", "99%", "max"]]
    resumen.columns = ["archivos", "mas_temprano", "mediana", "p90", "p99",
                       "mas_tarde"]

    print("Hora de generación del archivo del día siguiente, por año\n")
    print(resumen.round(2).to_string())

    tarde = vispera[vispera.hora >= 20]
    print(f"\npublicaciones después de las 20:00: {len(tarde)} de {len(vispera)} "
          f"({100 * len(tarde) / len(vispera):.1f}%)")
    print(f"archivos generados el mismo día de operación o después: "
          f"{int((sellos.anticipacion_dias <= 0).sum())}")
    print(f"margen mínimo contra el cierre de ofertas de las 10:00 del día de "
          f"generación: {vispera.hora.min() - CIERRE_OFERTAS:.1f} h")

    RESULTS.mkdir(exist_ok=True)
    resumen.round(3).to_csv(RESULTS / "publication_hours_by_year.csv")
    print(f"\nescrito {RESULTS / 'publication_hours_by_year.csv'}")


if __name__ == "__main__":
    main()
