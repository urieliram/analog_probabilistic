"""
Mapa de las 25 zonas de carga del estudio sobre el contorno de México.

Cada zona se coloca en la ciudad que le da nombre, con coordenadas aproximadas; las
tres zonas extremas fijadas de antemano (``selected_zones.json``) van en rojo. El
contorno es un GeoJSON público del país (``data/mexico.geojson``). Baja California no
tiene zonas en el estudio porque es un sistema eléctrico aparte.

    python experiments/figura_mapa_zonas.py
    FIGURAS_IDIOMA=en python experiments/figura_mapa_zonas.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.guardar import guarda  # noqa: E402
from experiments.idioma import carpeta, t  # noqa: E402
from experiments.protocol import RESULTS, load_selected_zones  # noqa: E402

FIGURAS = carpeta(RESULTS.parent / "figuras")
CONTORNO = RESULTS.parent / "data" / "mexico.geojson"

## la ciudad que da nombre a cada zona: (latitud, longitud), aproximadas
CIUDADES = {
    "caborca": (30.72, -112.16), "hermosillo": (29.07, -110.95), "obregon": (27.49, -109.94),
    "navojoa": (27.07, -109.44), "culiacan": (24.80, -107.39), "mazatlan": (23.25, -106.41),
    "chihuahua": (28.63, -106.07), "durango": (24.02, -104.66), "fresnillo": (23.17, -102.87),
    "nuevo_laredo": (27.49, -99.51), "victoria": (23.74, -99.14), "tampico": (22.26, -97.87),
    "tepic_vallarta": (21.51, -104.89), "zamora": (19.99, -102.28), "ixmiquilpan": (20.48, -99.22),
    "centro_oriente": (19.42, -98.95), "centro_sur": (19.25, -99.20), "tecamachalco": (18.88, -97.73),
    "teziutlan": (19.82, -97.36), "poza_rica": (20.53, -97.46), "veracruz": (19.19, -96.14),
    "coatzacoalcos": (18.15, -94.42), "villahermosa": (17.99, -92.93), "merida": (20.97, -89.62),
    "chilpancingo": (17.55, -99.50),
}
NOMBRES = {
    "obregon": "Obregón", "culiacan": "Culiacán", "mazatlan": "Mazatlán", "nuevo_laredo": "Nuevo Laredo",
    "tepic_vallarta": "Tepic-Vallarta", "centro_oriente": "Centro Oriente", "centro_sur": "Centro Sur",
    "teziutlan": "Teziutlán", "poza_rica": "Poza Rica", "merida": "Mérida",
}
## por que es obligatoria cada zona extrema, en los dos idiomas
RAZON = {"spike_rate": ("más picos", "most spikes"),
         "congestion_share": ("más congestión", "most congestion"),
         "negative_rate": ("más precios negativos", "most negative prices")}
## donde va la etiqueta respecto del punto, para que no se encimen
LADO = {"navojoa": (0.3, -0.45), "obregon": (0.3, 0.15), "hermosillo": (0.3, 0.1),
        "centro_sur": (-0.3, -0.3), "centro_oriente": (-0.3, 0.4), "tecamachalco": (0.3, -0.35),
        "teziutlan": (0.3, 0.45), "poza_rica": (0.3, 0.25), "ixmiquilpan": (-0.3, 0.3),
        "chilpancingo": (0.3, -0.3), "zamora": (-0.3, -0.45), "tepic_vallarta": (-0.3, -0.45),
        "victoria": (0.3, 0.1), "tampico": (0.3, -0.1), "mazatlan": (-0.3, -0.4),
        "culiacan": (0.3, 0.1), "veracruz": (0.3, 0.15), "coatzacoalcos": (-0.3, -0.5),
        "villahermosa": (0.3, -0.45), "merida": (0.3, 0.2)}


def main():
    sel = load_selected_zones()
    zonas, extremas = sel["zonas"], sel["obligatorias"]
    faltan = set(zonas) - set(CIUDADES)
    assert not faltan, f"sin coordenadas: {faltan}"
    geo = json.load(open(CONTORNO))["features"][0]["geometry"]
    anillos = ([geo["coordinates"][0]] if geo["type"] == "Polygon"
               else [p[0] for p in geo["coordinates"]])

    fig, ax = plt.subplots(figsize=(11, 7.5))
    for anillo in anillos:
        ax.add_patch(Polygon(anillo, closed=True, facecolor="#f3f1ec", edgecolor="#8d8d8d",
                             lw=0.8))
    for zona in zonas:
        lat, lon = CIUDADES[zona]
        extrema = zona in extremas
        ax.plot(lon, lat, "o", ms=8 if extrema else 6,
                color="#c62828" if extrema else "#1565c0", zorder=3)
        dx, dy = LADO.get(zona, (0.3, 0.1))
        nombre = NOMBRES.get(zona, zona.replace("_", " ").title())
        if extrema:
            nombre += "\n(" + t(*RAZON[extremas[zona]]) + ")"
        ax.text(lon + dx, lat + dy, nombre, fontsize=8.5, ha="left" if dx > 0 else "right",
                va="center", color="#8e0000" if extrema else "#0d2b55", zorder=4)
    ax.set_xlim(-118, -86)
    ax.set_ylim(14, 33)
    ax.set_aspect(1 / 0.93)
    ax.set_xticks([])
    ax.set_yticks([])
    for borde in ax.spines.values():
        borde.set_visible(False)
    ax.set_title(t(f"Las {len(zonas)} zonas de carga del estudio, sobre las "
                   f"{sel['zonas_evaluadas']} del sistema interconectado",
                   f"The {len(zonas)} load zones of the study, out of the "
                   f"{sel['zonas_evaluadas']} of the interconnected system"),
                 loc="left", fontsize=12, fontweight="bold")
    ax.text(-118, 14.3, t(
        "Cada punto es la ciudad que da nombre a la zona. En rojo, las tres zonas extremas "
        "fijadas de antemano; las otras 22 son las centrales de cada grupo de zonas "
        "parecidas.\nBaja California no tiene zonas en el estudio: es un sistema eléctrico "
        "aparte.",
        "Each dot is the city that names the zone. In red, the three extreme zones fixed in "
        "advance; the other 22 are the central ones of each group of similar zones.\n"
        "Baja California has no zones in the study: it is a separate power system."),
        fontsize=8.5, color="#555555", va="bottom")
    fig.tight_layout()
    FIGURAS.mkdir(exist_ok=True)
    ruta = FIGURAS / "figura_mapa_zonas.png"
    guarda(fig, ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("escrito", ruta)


if __name__ == "__main__":
    main()
