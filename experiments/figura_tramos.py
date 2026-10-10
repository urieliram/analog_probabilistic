"""
Esquema de los tramos del estudio: qué parte de la serie se usó para qué.

Arriba, la línea de los años: historia previa, selección, prueba, tramo posterior y
prueba prospectiva, más la réplica de la malla de selección. Abajo, lo que usa cada
día de pronóstico: no hay un tramo fijo de entrenamiento, todo se ajusta con ventanas
que ruedan y que sólo miran el pasado.

Las fechas salen de ``protocol.py``, de ``grid_sweep.py`` (la réplica) y de
``confirmacion_2026.py`` (el inicio prospectivo); los largos de las ventanas, de los
módulos que las usan.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.benchmarks.lear import DIAS_CALIBRACION  # noqa: E402
from experiments.calibra_escenarios import HISTORIA_MINIMA  # noqa: E402
from experiments.confirmacion_2026 import INICIO_PROSPECTIVO  # noqa: E402
from experiments.conformal import VENTANA_RODANTE  # noqa: E402
from experiments.escenarios_de_errores_calendario import VENTANA_ERRORES  # noqa: E402
from experiments.protocol import (INSPECTED_END, INSPECTED_START,  # noqa: E402
                                  RESULTS, SEARCH_YEARS, SELECTION_END,
                                  SELECTION_START, TEST_END, TEST_START)

FIGURAS = RESULTS.parent / "figuras"
INICIO_DATOS, FIN_DATOS = "2018-04-04", "2026-10-06"
REPLICA = ("2022-04-01", "2023-03-31")      # grid_sweep.py, ventana "limpio"

GRIS, NARANJA, AZUL, VERDE, MORADO = "#bdbdbd", "#ef6c00", "#1565c0", "#00897b", "#6a1b9a"


def _fecha(texto):
    return mdates.date2num(pd.Timestamp(texto))


def _bloque(eje, inicio, fin, y, alto, color, texto="", borde=None, rayado=None,
            discontinuo=False, color_texto="white", tam=10):
    x0, x1 = _fecha(inicio), _fecha(fin)
    eje.add_patch(Rectangle((x0, y - alto / 2), x1 - x0, alto, facecolor=color,
                            edgecolor=borde or color, hatch=rayado, linewidth=1.5,
                            linestyle="--" if discontinuo else "-"))
    if texto:
        eje.text((x0 + x1) / 2, y, texto, ha="center", va="center", fontsize=tam,
                 color=color_texto, fontweight="bold")
    return x0, x1


def linea_de_tiempo(eje):
    y, alto = 2.0, 0.5
    ## los precios publicados, como un riel encima de los tramos
    x0, x1 = _fecha(INICIO_DATOS), _fecha(FIN_DATOS)
    eje.plot([x0, x1], [2.62, 2.62], color="#616161", lw=3, solid_capstyle="round")
    eje.plot([x1, _fecha("2027-03-31")], [2.62, 2.62], color="#9e9e9e", lw=2, ls=":")
    eje.text(x0, 2.78, "precios horarios publicados, abril 2018 a octubre 2026",
             fontsize=9, color="#424242")
    ## los tramos: dentro, sólo el nombre
    fin_historia = pd.Timestamp(SELECTION_START) - pd.Timedelta(days=1)
    h0, h1 = _bloque(eje, INICIO_DATOS, fin_historia, y, alto, "#eeeeee", "Historia",
                     borde=GRIS, color_texto="#616161")
    s0, s1 = _bloque(eje, SELECTION_START, SELECTION_END, y, alto, NARANJA, "Selección")
    p0, p1 = _bloque(eje, TEST_START, TEST_END, y, alto, AZUL,
                     "Prueba · abril 2021 a febrero 2026")
    i0, i1 = _bloque(eje, INSPECTED_START, INSPECTED_END, y, alto, VERDE)
    f0, f1 = _bloque(eje, INICIO_PROSPECTIVO, "2027-03-31", y, alto, "white",
                     borde=MORADO, discontinuo=True)
    ## debajo de los tramos anchos, para qué sirvieron
    debajo = 1.55
    eje.text((h0 + h1) / 2, debajo, "sólo historia\npara buscar análogos",
             ha="center", va="top", fontsize=8.6, color="#616161", linespacing=1.3)
    eje.text((s0 + s1) / 2, debajo, "validación: se eligió\nla configuración\n"
             "del análogo\n(valle de precios\nde la pandemia)", ha="center", va="top",
             fontsize=8.6, color=NARANJA, linespacing=1.3)
    ## las zonas se eligieron con la historia y la selección juntas
    eje.plot([h0, h0, s1, s1], [0.86, 0.8, 0.8, 0.86], color="#616161", lw=1)
    eje.text((h0 + s1) / 2, 0.74, "las 25 zonas se eligieron con estos\ntres años, antes "
             "de la prueba", ha="center", va="top", fontsize=8.4, color="#616161",
             linespacing=1.25)
    eje.text((p0 + p1) / 2, debajo, "todos los resultados del artículo\naquí se "
             "diagnosticaron el calendario y la forma de la calibración", ha="center",
             va="top", fontsize=8.6, color=AZUL, linespacing=1.3)
    ## la réplica de la malla, abajo de la prueba
    _, r1 = _bloque(eje, *REPLICA, 0.62, 0.26, "white", borde=AZUL, rayado="///")
    eje.text(r1 + 25, 0.62, "réplica de la selección: comprobación dentro\n"
             "de la prueba, no se usó para elegir", va="center", fontsize=8.4,
             color=AZUL, linespacing=1.25)
    ## los tramos angostos se explican a la derecha, con una línea guía
    derecha = _fecha("2027-08-01")
    for (a, b), yy, color, titulo, texto in [
            ((i0, i1), 2.32, VERDE, "Posterior · marzo a septiembre 2026",
             "confirmación; ya lo recorrió una versión anterior"),
            ((f0, f1), 1.42, MORADO, "Prospectivo · desde el 11 de octubre 2026",
             "intacto: días que aún no existen;\nhipótesis registradas el 9 de octubre")]:
        borde_y = y + alto / 2 if yy > y else y - alto / 2
        eje.plot([(a + b) / 2, derecha - 15], [borde_y, yy], color=color, lw=1.1)
        eje.text(derecha, yy, titulo, fontsize=9.2, color=color, fontweight="bold",
                 va="bottom")
        eje.text(derecha, yy - 0.04, texto, fontsize=8.6, color=color, va="top",
                 linespacing=1.3)
    eje.text(_fecha("2023-10-01"), 0.1, "No se apartó ningún dato: la pandemia sólo coincide con la "
             "selección.", fontsize=8.6, color="#616161", style="italic")
    eje.set_ylim(0.0, 3.0)
    eje.set_xlim(_fecha("2018-01-01"), _fecha("2030-01-01"))
    eje.set_xticks([_fecha(f"{a}-01-01") for a in range(2018, 2028)])
    eje.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    eje.spines["bottom"].set_bounds(_fecha("2018-01-01"), _fecha("2027-06-01"))
    eje.set_yticks([])
    for lado in ("left", "right", "top"):
        eje.spines[lado].set_visible(False)
    eje.tick_params(axis="x", labelsize=10)
    eje.set_title("Qué parte de la serie se usó para qué", fontsize=13, loc="left",
                  pad=8, fontweight="bold")


def un_dia(eje):
    """Las ventanas que usa cada día de pronóstico; no está a escala."""
    filas = [
        (f"{SEARCH_YEARS} años", 9.0, GRIS, "banco donde el análogo busca días parecidos; "
         f"LEAR se ajusta con {DIAS_CALIBRACION} días, cada semana"),
        (f"{VENTANA_RODANTE} días", 4.2, AZUL, "calibración de los escenarios por "
         f"nivel y bloque de horas (mínimo {HISTORIA_MINIMA} días)"),
        (f"{VENTANA_ERRORES} errores", 2.6, VERDE, "escenarios de errores: los últimos "
         "del mismo tipo de día"),
    ]
    for i, (largo, ancho, color, texto) in enumerate(filas):
        y = 2.3 - i * 0.8
        eje.barh(y, ancho, left=-ancho, height=0.4, color=color)
        eje.text(-0.15, y, largo, va="center", ha="right", fontsize=9.5, color="white",
                 fontweight="bold")
        eje.text(-ancho - 0.25, y, texto, va="center", ha="right", fontsize=9,
                 color="#424242")
    eje.plot([0, 0], [0.45, 2.75], color="black", lw=1.6)
    eje.text(0, 2.82, "decisión, 10:00", ha="center", va="bottom", fontsize=9.5,
             fontweight="bold")
    eje.annotate("", xy=(1.0, 1.5), xytext=(0.1, 1.5),
                 arrowprops=dict(arrowstyle="->", color=MORADO, lw=1.5))
    eje.barh(1.5, 3.2, left=1.1, height=0.62, color=MORADO)
    eje.text(2.7, 1.5, "día pronosticado\n24 horas", ha="center", va="center",
             fontsize=9, color="white", fontweight="bold", linespacing=1.2)
    eje.text(-21.5, 2.82, "no está a escala", fontsize=8.5, color="#9e9e9e",
             va="bottom")
    eje.text(-21.5, 0.2, "Cada día usa sólo su pasado: no hay un tramo fijo de "
             "entrenamiento, todo rueda con el día.\nLos modelos de fundación vienen "
             "preentrenados, sin ajustarse a esta serie.", fontsize=8.6,
             color="#616161", style="italic", va="center", linespacing=1.35)
    eje.set_xlim(-21.5, 4.6)
    eje.set_ylim(-0.1, 3.15)
    eje.axis("off")
    eje.set_title("Qué usa cada día de pronóstico", fontsize=13, loc="left", pad=6,
                  fontweight="bold")


def main():
    fig, (arriba, abajo) = plt.subplots(2, 1, figsize=(14, 8.2),
                                        gridspec_kw={"height_ratios": [1.5, 1]})
    linea_de_tiempo(arriba)
    un_dia(abajo)
    fig.tight_layout(h_pad=2.2)
    FIGURAS.mkdir(exist_ok=True)
    ruta = FIGURAS / "figura_tramos.png"
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("escrito", ruta)


if __name__ == "__main__":
    main()
