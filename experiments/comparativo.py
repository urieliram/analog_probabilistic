"""
El cuadro comparativo de todo el banco, en crudo y calibrado.

Junta lo que produjeron las corridas —las variantes del análogo, los comparativos
ingenuos, LEAR y lo que haya— y responde tres preguntas con la misma regla para todos:

1. **Quién pronostica mejor** por pérdida pinball media y CRPS del ensamble, con la
   prueba pareada por origen que corresponde.
2. **Quién miente sobre su propia incertidumbre**: cobertura prometida contra entregada,
   y cuánto le cuesta dejar de mentir al pasar por la capa conformal.
3. **Quién subestima los picos**, que es la pregunta que importa para cubrirse: cuántas
   veces el precio se escapa por arriba del intervalo en el cuarto de días más movidos.

Escribe un resumen en palabras llanas, porque un cuadro de números sin su lectura se
usa mal.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.protocol import RESULTS, load_selected_zones  # noqa: E402
from experiments.scores import diebold_mariano  # noqa: E402
from experiments.series import load_series  # noqa: E402

FUENTES = ("picos_prueba_conformal.csv", "bench_prueba_conformal.csv",
           "lear_prueba_conformal.csv", "forecasts_prueba_conformal.csv")
REFERENCIA = "Analog"


def carga() -> pd.DataFrame:
    partes = []
    for nombre in FUENTES:
        ruta = RESULTS / nombre
        if not ruta.exists():
            print(f"  (falta {nombre}: se omite)")
            continue
        t = pd.read_csv(ruta, parse_dates=["origin"])
        t["fuente"] = nombre.replace("_conformal.csv", "")
        partes.append(t)
    if not partes:
        raise SystemExit("no hay ninguna corrida calibrada que comparar")
    t = pd.concat(partes, ignore_index=True)
    ## el análogo crudo aparece en más de una corrida: se queda una sola copia
    return t.drop_duplicates(subset=["zone", "origin", "method", "step"]
                             if "step" in t.columns else
                             ["zone", "origin", "method"])


def movimiento_real(zonas: list) -> pd.DataFrame:
    """Cuánto recorrió de verdad el día que se pronosticó, por zona y origen."""
    filas = []
    for z in zonas:
        s = load_series("pml", z)
        d = s.resample("D")
        r = (d.max() - d.min()).rename("rango")
        r.index = r.index - pd.Timedelta(hours=1)   ## el origen es la hora 23 anterior
        filas.append(r.to_frame().assign(zone=z))
    return pd.concat(filas).reset_index().rename(columns={"ds": "origin"})


def main() -> None:
    t = carga()
    ## El intervalo de 80% es la columna común del banco, no el de 90%. TiRex y
    ## TimesFM se detienen en el cuantil 0.9, así que su intervalo central más ancho de
    ## fábrica es el de 80%; al 90% unos modelos llegarían por cuantil propio y otros
    ## por extrapolación, y la misma columna mediría dos cosas distintas. El 90% y el
    ## 95% se reportan aparte, sólo para los que los emiten nativos.
    t["escapa_abajo"] = t["hit_0.1"]
    t["escapa_arriba"] = 1 - t["hit_0.9"]
    t["escapa_abajo_90"] = t["hit_0.05"]
    t["escapa_arriba_90"] = 1 - t["hit_0.95"]
    t["calibrado"] = t.method.str.contains(r"\+")
    t["base"] = t.method.str.split("+").str[0]
    t["capa"] = np.where(t.calibrado, t.method.str.split("+").str[1], "crudo")

    print(f"{t.method.nunique()} métodos · {t.zone.nunique()} zonas · "
          f"{t.origin.nunique()} orígenes · {len(t):,d} pronósticos evaluados\n")

    ## columna común: el intervalo de 80%, nativo en todos los métodos del banco
    columnas = ["mean_pinball", "crps", "cover80", "escapa_abajo", "escapa_arriba",
                "sharp80", "winkler80", "mae", "reliability"]
    ## sólo para los que llegan al 0.05 y 0.95 de fábrica
    columnas_90 = ["cover90", "escapa_abajo_90", "escapa_arriba_90", "sharp90",
                   "winkler90"]

    print("=" * 78)
    print("1. EN CRUDO: lo que cada método entrega de fábrica")
    print("=" * 78)
    crudo = t[~t.calibrado].groupby("method")[columnas].mean()
    print(crudo.sort_values("mean_pinball").round(3).to_string())

    print("\n" + "=" * 78)
    print("2. CALIBRADO: todos por la misma capa conformal")
    print("=" * 78)
    for capa in sorted(t[t.calibrado].capa.unique()):
        print(f"\n--- capa: {capa} ---")
        sub = t[t.capa == capa].groupby("base")[columnas].mean()
        print(sub.sort_values("mean_pinball").round(3).to_string())

    print("\n" + "=" * 78)
    print("3. LA PREGUNTA DE COBERTURA: cuánto cuesta dejar de mentir")
    print("=" * 78)
    print("se prometió cobertura 0.90\n")
    piv = t.pivot_table(index="base", columns="capa", values="cover90")
    anchos = t.pivot_table(index="base", columns="capa", values="sharp90")
    for col in piv.columns:
        if col != "crudo" and "crudo" in anchos.columns:
            piv[f"ancho_{col}_vs_crudo"] = (anchos[col] / anchos["crudo"]).round(2)
    print(piv.round(3).to_string())

    print("\n" + "=" * 78)
    print("4. LA PREGUNTA CARA: subestimación de picos")
    print("=" * 78)
    rangos = movimiento_real(load_selected_zones()["zonas"])
    j = t.merge(rangos, on=["zone", "origin"], how="inner").dropna(subset=["rango"])
    j["cuarto"] = pd.qcut(j.rango, 4, labels=["tranquilo+", "tranquilo",
                                              "movido", "movido+"])
    peor = j[j.cuarto == "movido+"].groupby("method")[
        ["escapa_arriba", "escapa_abajo", "cover80", "sharp80",
         "escapa_arriba_90", "cover90"]].mean()
    print("en el cuarto de días MÁS MOVIDOS. En el intervalo común de 80% se prometió")
    print("0.10 de escape por lado; la columna _90 es el 0.05 del intervalo de 90%\n")
    print(peor.sort_values("escapa_arriba").round(4).to_string())

    print("\n--- cobertura del 80% por cuarto de movimiento, peor desvío del 0.80 ---")
    p2 = j.pivot_table(index="method", columns="cuarto", values="cover80",
                       observed=True)
    p2["peor_desvio"] = (p2 - 0.80).abs().max(axis=1)
    print(p2.round(3).sort_values("peor_desvio").to_string())

    print("\n" + "=" * 78)
    print(f"5. PRUEBAS PAREADAS contra {REFERENCIA}, por origen")
    print("=" * 78)
    print("un estadístico negativo favorece al método de la fila\n")
    if REFERENCIA in set(t.method):
        base = t[t.method == REFERENCIA].groupby("origin")["mean_pinball"].mean()
        filas = []
        for metodo in sorted(set(t.method) - {REFERENCIA}):
            otro = t[t.method == metodo].groupby("origin")["mean_pinball"].mean()
            comun = base.index.intersection(otro.index)
            if len(comun) < 30:
                continue
            dif = (otro.loc[comun] - base.loc[comun]).sort_index().to_numpy()
            est, _ = diebold_mariano(dif)
            filas.append({"metodo": metodo, "diferencia": dif.mean(),
                          "t": est, "origenes": len(comun),
                          "favorece": metodo if dif.mean() < 0 else REFERENCIA})
        print(pd.DataFrame(filas).sort_values("diferencia").round(3)
              .to_string(index=False))
    else:
        print(f"  (no hay método {REFERENCIA} en las corridas cargadas)")

    print("\n" + "=" * 78)
    print("6. EL 90%, sólo para los métodos que llegan al 0.05 y 0.95 de fábrica")
    print("=" * 78)
    print("el análogo llega porque sus cuantiles salen del ensamble; TiRex y TimesFM")
    print("se detienen en 0.9 y aquí no aparecerían\n")
    print(t.groupby("method")[columnas_90].mean().round(3).to_string())

    salida = RESULTS / "comparativo_banco.csv"
    t.groupby("method")[columnas + columnas_90].mean().round(6).to_csv(salida)
    peor.round(6).to_csv(RESULTS / "comparativo_picos.csv")
    print(f"\nescrito {salida.name} y comparativo_picos.csv")


if __name__ == "__main__":
    main()
