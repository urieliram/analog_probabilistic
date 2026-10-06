"""
Reduce los puntajes de una corrida a las dos tablas que el artículo necesita.

El archivo por pronóstico pesa decenas de megabytes —ochenta y nueve mil filas por
corrida— y regenerarlo cuesta media hora de máquina, así que no cabe en el
repositorio ni conviene perderlo. Lo que sí cabe, y es lo que sostiene cada número
publicado, son dos agregados de unos pocos miles de filas:

- **por origen**, promediado sobre las zonas: es la serie sobre la que corre la
  prueba de Diebold-Mariano, y promediar antes de probar es obligatorio porque las
  zonas de un mismo origen comparten el día;
- **por zona y año**, que es lo que muestra si una diferencia es sistemática o la
  cargan dos zonas y un año malo.

El archivo completo va al depósito con identificador permanente, junto al panel.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.protocol import RESULTS  # noqa: E402

PUNTAJES = ["mean_pinball", "crps", "reliability", "mae", "rmse",
            "cover80", "sharp80", "winkler80", "cover90", "sharp90",
            "winkler90", "pinball95", "wis", "seconds"]


def resumir(ruta: Path) -> dict:
    puntajes = pd.read_csv(ruta, parse_dates=["origin"])
    puntajes["anio"] = puntajes.origin.dt.year
    columnas = [c for c in PUNTAJES if c in puntajes.columns]

    por_origen = puntajes.groupby(["method", "origin"])[columnas].mean()
    por_zona_anio = puntajes.groupby(["method", "zone", "anio"])[columnas].mean()

    raiz = ruta.name.replace("_puntajes_", "_").replace(".csv", "")
    salidas = {}
    for nombre, tabla in [("por_origen", por_origen),
                          ("por_zona_anio", por_zona_anio)]:
        destino = RESULTS / f"{raiz}_{nombre}.csv"
        tabla.round(6).to_csv(destino)
        salidas[destino.name] = len(tabla)
    return salidas


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("uso: resumir_corrida.py <puntajes.csv> [...]")
    for argumento in sys.argv[1:]:
        for nombre, filas in resumir(Path(argumento)).items():
            print(f"{nombre}: {filas:,d} filas")


if __name__ == "__main__":
    main()
