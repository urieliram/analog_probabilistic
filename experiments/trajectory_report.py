"""
Mide la dependencia entre horas, que es lo único que el método tiene y los demás no.

Cada miembro del ensamble es un camino completo de veinticuatro horas, así que el
conjunto carga la relación entre una hora y la siguiente. Afirmarlo sin medirlo no
vale: aquí se mide con el puntaje de energía y el de variograma, y con tres eventos
de camino que ningún conjunto de cuantiles marginales puede contestar sin un
supuesto extra.

Los umbrales de los eventos salen del tramo de selección, nunca del de prueba.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.forecast_store import (load_forecasts,  # noqa: E402
                                        load_members, member_columns)
from experiments.protocol import RESULTS  # noqa: E402
from experiments.trajectory_scores import energy_score, variogram_score  # noqa: E402

UMBRAL_RELATIVO = 3.0
HORAS_SEGUIDAS = 3


def eventos(observado: np.ndarray, miembros: np.ndarray,
            umbral: float) -> dict:
    """
    Probabilidades de camino y lo que de verdad ocurrió.

    El máximo del día y las horas seguidas por encima de un umbral no se contestan
    con cuantiles hora por hora: hacen falta los caminos.
    """
    def corrida_maxima(fila: np.ndarray) -> int:
        arriba = fila > umbral
        mejor = actual = 0
        for valor in arriba:
            actual = actual + 1 if valor else 0
            mejor = max(mejor, actual)
        return mejor

    return {
        "p_maximo": float((miembros.max(axis=1) > umbral).mean()),
        "y_maximo": float(observado.max() > umbral),
        "p_seguidas": float(np.mean([corrida_maxima(m) >= HORAS_SEGUIDAS
                                     for m in miembros])),
        "y_seguidas": float(corrida_maxima(observado) >= HORAS_SEGUIDAS),
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("uso: trajectory_report.py <corrida> [n_zonas]")
    ruta = Path(sys.argv[1])
    limite = int(sys.argv[2]) if len(sys.argv) > 2 else None

    pronosticos = load_forecasts(ruta)
    miembros = load_members(ruta)
    if miembros is None:
        raise SystemExit("esa corrida no trae miembros")
    columnas = member_columns(miembros)

    zonas = sorted(pronosticos.zone.unique())[:limite]
    umbrales = {z: UMBRAL_RELATIVO * pronosticos[pronosticos.zone == z]
                ["observed"].median() for z in zonas}

    filas = []
    for zona in zonas:
        bloque = pronosticos[pronosticos.zone == zona]
        bloque_m = miembros[miembros.zone == zona]
        for (origen, metodo), paso in bloque.groupby(["origin", "method"],
                                                     sort=False):
            muestra = bloque_m[(bloque_m.origin == origen)
                               & (bloque_m.method == metodo)]
            if muestra.empty:
                continue
            paso = paso.sort_values("step")
            ensamble = muestra.sort_values("step")[columnas].to_numpy().T
            observado = paso["observed"].to_numpy()
            filas.append({
                "zone": zona, "origin": origen, "method": metodo,
                "energia": energy_score(observado, ensamble),
                "variograma": variogram_score(observado, ensamble),
                **eventos(observado, ensamble, umbrales[zona]),
            })

    tabla = pd.DataFrame(filas)
    resumen = tabla.groupby("method")[["energia", "variograma"]].mean()
    print(f"{tabla.zone.nunique()} zonas · {len(tabla):,d} pronósticos\n")
    print(resumen.round(3).to_string())

    print("\neventos de camino: probabilidad que anuncia y frecuencia observada")
    for metodo, bloque in tabla.groupby("method"):
        print(f"  {metodo}")
        print(f"    máximo del día sobre el umbral: anuncia "
              f"{100*bloque.p_maximo.mean():5.1f}%, ocurre "
              f"{100*bloque.y_maximo.mean():5.1f}%")
        print(f"    tres horas seguidas encima:     anuncia "
              f"{100*bloque.p_seguidas.mean():5.1f}%, ocurre "
              f"{100*bloque.y_seguidas.mean():5.1f}%")
        brier = ((bloque.p_maximo - bloque.y_maximo) ** 2).mean()
        print(f"    puntaje de Brier del máximo: {brier:.4f}")

    tabla.to_csv(RESULTS / f"{ruta.stem}_trayectorias.csv", index=False)
    print(f"\nescrito {ruta.stem}_trayectorias.csv")


if __name__ == "__main__":
    main()
