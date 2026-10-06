"""
Elige el panel de zonas: cuántas y cuáles, con el número decidido por los datos.

Las 101 zonas del sistema no son 101 evidencias. Están unidas por la misma red —
unas radiales, otras eléctricamente vecinas— y muchas series son prácticamente la
misma. Pero el parecido hay que medirlo donde importa: sobre el precio crudo manda
el factor común del sistema, gas y demanda nacional, y agrupar ahí seleccionaría
por nivel de precio. Lo que hace difícil un pronóstico —congestión local, picos,
precios negativos— vive en lo que queda al quitar ese factor.

Procedimiento, fijado antes de correr:

  1. panel de las 101 zonas, sólo sobre el tramo de selección;
  2. estandarizar y quitar el primer componente;
  3. agrupamiento jerárquico sobre la correlación de los residuos, corte declarado;
  4. de cada grupo, su centro, más las zonas extremas declaradas de antemano.

El resultado se congela en results/selected_zones.json y NO se revisa a mano
después: mirar el resultado y añadir o quitar zonas porque «parecen difíciles» es
discrecionalidad después de ver los datos.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.protocol import (RESULTS, SELECTED_ZONES_FILE,  # noqa: E402
                                  SELECTION_END, SELECTION_START,
                                  ZONE_CLUSTER_THRESHOLD,
                                  ZONE_MANDATORY_EXTREMES)

PANEL_START = "2018-04-04"

ZONAS = Path(__file__).resolve().parents[1] / "data" / "zones"


def cargar_panel(inicio: str, fin: str, columna: str = "pml") -> pd.DataFrame:
    """Una columna de todas las zonas, en el tramo pedido."""
    series = {}
    for archivo in sorted(ZONAS.glob("*.csv.gz")):
        tabla = pd.read_csv(archivo, parse_dates=["ds"]).set_index("ds")
        series[archivo.name[: -len(".csv.gz")]] = tabla.loc[inicio:fin, columna]
    panel = pd.DataFrame(series)
    return panel.dropna(axis=1, thresh=int(0.98 * len(panel))).dropna()


def caracterizar(panel: pd.DataFrame, congestion: pd.DataFrame) -> pd.DataFrame:
    """
    Las seis características con que se describe cada zona en el artículo.

    La volatilidad se mide sobre diferencias y no sobre cambios porcentuales: hay
    zonas con precios cercanos a cero y negativos, donde un cambio porcentual no
    significa nada.
    """
    mediana = panel.median()
    return pd.DataFrame({
        "nivel": mediana,
        "volatilidad": panel.diff().std(),
        "spike_rate": (panel > 3 * mediana).mean(),
        "negative_rate": (panel < 0).mean(),
        "congestion_share": congestion.abs().mean() / panel.abs().mean(),
        "autocorrelacion_24": panel.apply(lambda s: s.autocorr(24)),
    })


def agrupar(panel: pd.DataFrame, umbral: float):
    """
    Agrupa las zonas por su parte idiosincrática.

    Devuelve el grupo de cada zona, el centro de cada grupo y cuánta varianza
    explica el factor común que se quitó.
    """
    estandar = (panel - panel.mean()) / panel.std()
    u, s, vt = np.linalg.svd(estandar.values, full_matrices=False)
    varianza_factor_comun = float(s[0] ** 2 / (s ** 2).sum())
    residuo = estandar.values - np.outer(u[:, 0] * s[0], vt[0])

    correlacion = np.corrcoef(residuo.T)
    distancia = squareform(np.clip(1 - correlacion, 0, 2), checks=False)
    enlace = linkage(distancia, method="average")
    grupos = fcluster(enlace, t=1 - umbral, criterion="distance")

    centros = {}
    for grupo in np.unique(grupos):
        miembros = np.where(grupos == grupo)[0]
        ## el centro es la zona con mayor parecido medio a las demás del grupo
        interno = correlacion[np.ix_(miembros, miembros)].mean(axis=1)
        centros[int(grupo)] = panel.columns[miembros[int(np.argmax(interno))]]

    return (pd.Series(grupos, index=panel.columns, name="grupo"), centros,
            varianza_factor_comun)


def main() -> None:
    ## El panel se caracteriza con TODO lo anterior al inicio de la prueba, no
    ## sólo con el año de selección: el tramo de selección cae en 2020, un periodo
    ## atípicamente homogéneo —el factor común llega al 89% contra el 70% de un año
    ## normal— y elegir ahí subestimaría la diversidad del mercado. Sigue sin mirar
    ## nada posterior al inicio de la prueba, que es la única regla que importa.
    panel = cargar_panel(PANEL_START, SELECTION_END)
    congestion = cargar_panel(PANEL_START, SELECTION_END, "congestion")
    congestion = congestion[panel.columns].loc[panel.index]
    print(f"caracterización sobre {PANEL_START} a {SELECTION_END}: "
          f"{panel.shape[0]} horas x {panel.shape[1]} zonas")

    rasgos = caracterizar(panel, congestion)
    grupos, centros, factor_comun = agrupar(panel, ZONE_CLUSTER_THRESHOLD)
    print(f"el factor común explica {100 * factor_comun:.1f}% de la varianza")
    print(f"corte en r > {ZONE_CLUSTER_THRESHOLD}: {len(centros)} grupos")

    ## las extremas entran por regla declarada de antemano, no por inspección
    obligatorias = {rasgos[columna].idxmax(): columna
                    for columna in ZONE_MANDATORY_EXTREMES
                    if columna in rasgos.columns}

    panel_elegido = sorted(set(centros.values()) | set(obligatorias))
    print(f"obligatorias por extremo: {obligatorias}")
    print(f"panel: {len(panel_elegido)} zonas")

    correspondencia = pd.DataFrame({
        "zona": panel.columns,
        "grupo": grupos.values,
        "representante": [centros[g] for g in grupos.values],
    }).sort_values(["grupo", "zona"])
    correspondencia["en_el_panel"] = correspondencia.zona.isin(panel_elegido)
    correspondencia = correspondencia.merge(
        rasgos.round(4), left_on="zona", right_index=True)

    ## las 101 zonas se quedan fuera del repositorio: son 107 MB que git guardaría
    ## para siempre en cada regeneración, y se reconstruyen con build_panel.py. Al
    ## repositorio va sólo el panel elegido, que basta para reproducir el artículo.
    destino = ZONAS.parent / "panel"
    destino.mkdir(exist_ok=True)
    for archivo in destino.glob("*.csv.gz"):
        archivo.unlink()
    for zona in panel_elegido:
        shutil.copy2(ZONAS / f"{zona}.csv.gz", destino / f"{zona}.csv.gz")
    print(f"panel copiado a {destino}")

    RESULTS.mkdir(exist_ok=True)
    correspondencia.to_csv(RESULTS / "zone_groups.csv", index=False)
    with open(SELECTED_ZONES_FILE, "w") as handle:
        json.dump({
            "zonas": panel_elegido,
            "n_zonas": len(panel_elegido),
            "n_grupos": len(centros),
            "umbral": ZONE_CLUSTER_THRESHOLD,
            "factor_comun": round(factor_comun, 4),
            "obligatorias": obligatorias,
            "tramo": [PANEL_START, SELECTION_END],
            "zonas_evaluadas": int(panel.shape[1]),
        }, handle, indent=2, ensure_ascii=False)
    print(f"\ncongelado en {SELECTED_ZONES_FILE.name}")


if __name__ == "__main__":
    main()
