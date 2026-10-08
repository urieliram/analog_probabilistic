"""
Escoger los días por lo que pasó contra escogerlos por lo que se sabía al decidir.

Medir la cobertura en los días que RESULTARON caros parece la manera natural de saber si
un método protege en los picos, y es una trampa: castiga a cualquier pronóstico que no
anuncie siempre precios altos, incluso al perfecto. Es el dilema del pronosticador
(Lerch, Thorarinsdottir, Ravazzolo y Gneiting, 2017). Este módulo hace tres cosas:

1. **Lo demuestra con un pronosticador perfecto simulado.** Conoce la distribución
   verdadera de cada día y anuncia su intervalo exacto de 90%. Su cobertura total es
   0.90, y aun así se «desploma» en el decil de días que resultaron más caros.
2. **Reproduce esa medida sobre los métodos reales**, para mostrar cuánto de lo que
   parecía un hallazgo es el mismo efecto.
3. **La reemplaza por una prueba legítima.** Los días se agrupan por el costo del último
   día que el comprador ya conocía al decidir: el día siguiente al de la decisión, cuyos
   precios se publicaron la tarde anterior. Esa agrupación sólo usa información
   disponible, así que un método bien calibrado debe cubrir 0.90 en cada grupo, y la
   pinball por grupo sigue siendo un puntaje propio.

Uso: python experiments/dilema.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.protocol import RESULTS  # noqa: E402
from experiments.scores import diebold_mariano  # noqa: E402

DETALLE = RESULTS / "costo_diario_por_dia.parquet"
PRINCIPAL = "Analog-mezcla"
RIVALES = ["LEAR", "t0-beta+errores", "PatchTST-FM+errores"]
MOSTRAR = [PRINCIPAL, "Analog-amplitud-libre", "Analog", "Analog-crudo",
           "Analog-mezcla+errores", "LEAR", "t0-beta+errores", "PatchTST-FM+errores",
           "Moirai"]


def pronosticador_perfecto(dias: int = 200_000, semilla: int = 7) -> pd.DataFrame:
    """
    Un pronosticador que conoce la distribución verdadera de cada día.

    El costo del día es lognormal con un nivel y una dispersión que cambian de un día a
    otro, como cambia el régimen de precios. El pronosticador anuncia el intervalo
    exacto del 5 al 95%: es perfecto por construcción, y su cobertura total es 0.90.
    """
    rng = np.random.default_rng(semilla)
    nivel = rng.normal(np.log(17_000), 0.45, dias)
    dispersion = rng.uniform(0.15, 0.45, dias)
    costo = np.exp(nivel + dispersion * rng.standard_normal(dias))
    bajo = np.exp(nivel - 1.645 * dispersion)
    alto = np.exp(nivel + 1.645 * dispersion)
    t = pd.DataFrame({"costo": costo, "dentro": (costo >= bajo) & (costo <= alto),
                      "techo": alto})
    t["decil"] = pd.qcut(t.costo, 10, labels=range(1, 11)).astype(int)
    return t


def con_dia_anterior(t: pd.DataFrame) -> pd.DataFrame:
    """Añade el costo del último día conocido al decidir: el objetivo del origen previo."""
    previo = (t[["zona", "origen", "costo_real"]].drop_duplicates(["zona", "origen"])
              .rename(columns={"costo_real": "costo_conocido"}))
    previo["origen"] = previo["origen"] + pd.Timedelta(days=1)
    return t.merge(previo, on=["zona", "origen"], how="inner")


def deciles(valores: pd.Series) -> pd.Series:
    cortes = valores.quantile(np.arange(0, 1.01, 0.1)).to_numpy()
    return pd.Series(np.clip(np.searchsorted(cortes[1:-1], valores) + 1, 1, 10),
                     index=valores.index)


def compara(t: pd.DataFrame, a: str, b: str, columna: str) -> dict:
    pa = t[t.metodo == a].set_index(["zona", "origen"])[columna]
    pb = t[t.metodo == b].set_index(["zona", "origen"])[columna]
    comun = pa.index.intersection(pb.index)
    ## Se SUMA por día de decisión, no se promedia: en un subconjunto de días (un decil)
    ## cada día trae un número distinto de zonas, y promediar le daría a un día con una
    ## zona el mismo peso que a uno con veinte. Con sumas, el porcentaje coincide con el
    ## de los promedios simples sobre las mismas zonas y días.
    dif = (pa.loc[comun] - pb.loc[comun]).groupby(level="origen").sum().sort_index()
    base = pb.loc[comun].groupby(level="origen").sum().reindex(dif.index)
    estadistico, _ = diebold_mariano(dif.to_numpy())
    media = float(dif.mean())
    error = abs(media / estadistico) if estadistico else np.nan
    nivel = float(base.mean())
    return {"rival": b, "zona_dias": len(comun), "dias": len(dif),
            "pct": 100 * media / nivel, "t": estadistico,
            "pct_bajo": 100 * (media - 1.96 * error) / nivel,
            "pct_alto": 100 * (media + 1.96 * error) / nivel}


def main() -> None:
    pub = RESULTS / "publicacion"
    pub.mkdir(exist_ok=True)

    print("=== 1. un pronosticador PERFECTO, medido por decil de lo que pasó ===")
    p = pronosticador_perfecto()
    print(f"cobertura total: {p.dentro.mean():.3f} (anunció 0.90)")
    cob = p.groupby("decil").dentro.mean()
    print("cobertura por decil del costo ocurrido:")
    print(cob.round(3).to_string())
    cob.round(4).to_csv(pub / "cuadro_dilema_perfecto.csv", header=["cobertura_90"])

    t = pd.read_parquet(DETALLE)
    t = con_dia_anterior(t)
    t["decil_ocurrido"] = deciles(t.costo_real)
    t["decil_conocido"] = deciles(t.costo_conocido)

    print("\n=== 2. los métodos reales: cobertura 90 por decil de lo que PASÓ ===")
    print("(selección por el resultado: el dilema del pronosticador)")
    a = t[t.metodo.isin(MOSTRAR)].pivot_table(index="metodo", columns="decil_ocurrido",
                                              values="dentro_90")
    print(a.loc[MOSTRAR].round(3).to_string())

    print("\n=== 3. cobertura 90 por decil del costo del día YA CONOCIDO al decidir ===")
    print("(sólo información disponible: un método calibrado debe dar 0.90 en cada uno)")
    b = t[t.metodo.isin(MOSTRAR)].pivot_table(index="metodo", columns="decil_conocido",
                                              values="dentro_90")
    print(b.loc[MOSTRAR].round(3).to_string())
    b.round(4).to_csv(pub / "cuadro_cobertura_por_dia_conocido.csv")
    a.round(4).to_csv(pub / "cuadro_cobertura_por_dia_ocurrido.csv")

    print("\n=== pinball 0.95 por decil del día conocido (menos es mejor) ===")
    c = t[t.metodo.isin(MOSTRAR)].pivot_table(index="metodo", columns="decil_conocido",
                                              values="pin_0.95")
    print(c.loc[MOSTRAR].round(0).to_string())
    c.round(2).to_csv(pub / "cuadro_pinball95_por_dia_conocido.csv")

    print("\n=== Analog-mezcla contra rivales cuando el día conocido fue caro ===")
    filas = []
    for grupo, sub in [("deciles 1-9 del día conocido", t[t.decil_conocido <= 9]),
                       ("decil 10 del día conocido", t[t.decil_conocido == 10]),
                       ("todos los días", t)]:
        for tau in ["0.05", "0.5", "0.9", "0.95"]:
            for rival in RIVALES:
                r = compara(sub, PRINCIPAL, rival, f"pin_{tau}")
                r.update({"grupo": grupo, "tau": tau})
                filas.append(r)
    r = pd.DataFrame(filas)
    r.round(4).to_csv(pub / "cuadro_dia_conocido_pruebas.csv", index=False)
    for (grupo, tau), g in r.groupby(["grupo", "tau"], sort=False):
        celdas = "  ".join(f"{x.rival}: {x.pct:+.1f}% (t={x.t:+.1f})"
                           for x in g.itertuples())
        print(f"  {grupo:30s} tau={tau:5s} {celdas}")


if __name__ == "__main__":
    main()
