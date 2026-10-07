"""
¿Los miembros de un ensamble son días posibles, o sorteos hora por hora?

La pregunta importa porque un comprador de energía no paga hora por hora: paga el día.
El costo de un día depende de la forma completa —si el pico de la tarde coincide con el
consumo alto o no— y eso un pronóstico que sólo acierta las distribuciones por hora no lo
puede decir. Un método que entrega caminos coherentes dice algo que uno de cuantiles
marginales no dice, aunque los dos tengan la misma cobertura.

La medida es simple: se quita el perfil medio del ensamble y se mide la correlación entre
horas vecinas DENTRO de cada camino. Si cada miembro es un día posible, lo que lo aparta
del perfil medio persiste de una hora a la siguiente y la correlación es alta. Si cada
miembro es un sorteo independiente por hora, la correlación es cero.

Se incluye un sorteo deliberadamente independiente como testigo: cualquier método cuya
coherencia se parezca a ese testigo no tiene trayectorias, tenga o no un atributo que se
llame «muestras».
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.forecast_store import (load_members,  # noqa: E402
                                        member_columns, member_matrix)
from experiments.protocol import RESULTS  # noqa: E402


def coherencia(miembros: np.ndarray) -> dict:
    """
    Cuánto persiste de una hora a la siguiente lo que aparta a un camino del promedio.

    ``miembros`` tiene una fila por miembro y una columna por paso del horizonte.
    """
    m = np.asarray(miembros, dtype=float)
    if m.shape[0] < 2 or m.shape[1] < 3:
        return {"corr_global": float("nan"), "corr_por_camino": float("nan")}
    centrado = m - m.mean(axis=0, keepdims=True)
    a, b = centrado[:, :-1].ravel(), centrado[:, 1:].ravel()
    global_ = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else float("nan")
    por_camino = [float(np.corrcoef(c[:-1], c[1:])[0, 1])
                  for c in centrado if np.std(c[:-1]) and np.std(c[1:])]
    return {"corr_global": global_,
            "corr_por_camino": float(np.mean(por_camino)) if por_camino else float("nan")}


def testigo_independiente(miembros: np.ndarray, semilla: int = 0) -> dict:
    """
    Un ensamble con las MISMAS distribuciones por hora y las horas desatadas.

    Se permuta cada columna por separado: las distribuciones marginales quedan idénticas y
    toda la dependencia entre horas se destruye. Es el punto de comparación justo.
    """
    rng = np.random.default_rng(semilla)
    m = np.asarray(miembros, dtype=float).copy()
    for columna in range(m.shape[1]):
        rng.shuffle(m[:, columna])
    return coherencia(m)


def de_una_corrida(ruta: Path, origenes: int = 200) -> pd.DataFrame:
    """Coherencia media de cada método de una corrida, sobre una muestra de orígenes."""
    miembros = load_members(ruta)
    if miembros is None:
        return pd.DataFrame()
    columnas = member_columns(miembros)
    filas = []
    for (zona, metodo), bloque in miembros.groupby(["zone", "method"], sort=False):
        for origen, pron in list(bloque.groupby("origin", sort=False))[:origenes]:
            m = member_matrix(pron.sort_values("step"), columnas)
            if len(m) < 2:
                continue
            fila = {"metodo": metodo, "zona": zona, "miembros": len(m)}
            fila.update(coherencia(m))
            fila.update({f"testigo_{k}": v
                         for k, v in testigo_independiente(m).items()})
            filas.append(fila)
    return pd.DataFrame(filas)


def main() -> None:
    corridas = [p for p in sys.argv[1:]] or sorted(
        str(p) for p in RESULTS.glob("*_members") if p.is_dir())
    partes = []
    for corrida in corridas:
        base = Path(str(corrida).replace("_members", ""))
        t = de_una_corrida(base)
        if not t.empty:
            partes.append(t)
            print(f"  {base.name}: {t.metodo.nunique()} métodos, {len(t)} pronósticos")
    if not partes:
        raise SystemExit("ninguna corrida con miembros")

    t = pd.concat(partes, ignore_index=True)
    resumen = t.groupby("metodo")[["miembros", "corr_global", "corr_por_camino",
                                   "testigo_corr_global"]].mean()
    resumen["sobre_el_testigo"] = (resumen.corr_por_camino
                                   - resumen.testigo_corr_global)
    print("\n=== ¿cada miembro es un día posible? ===")
    print("corr_por_camino alto = sí; igual al testigo = no, son sorteos por hora\n")
    print(resumen.sort_values("corr_por_camino", ascending=False).round(4).to_string())
    salida = RESULTS / "coherencia_trayectorias.csv"
    resumen.round(6).to_csv(salida)
    print(f"\nescrito {salida.name}")


if __name__ == "__main__":
    main()
