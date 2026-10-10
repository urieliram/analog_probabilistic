"""
Tres figuras del artículo que resumen cuadros, recalculadas desde los resultados guardados.

Al principio se dibujaron con código suelto, y la del error por día de la semana tenía
los números escritos a mano. Aquí cada número sale de los archivos de resultados, de
modo que cualquiera puede rehacerlas:

1. ``figura_curva_tau_calibrada.png``: Analog-mix contra LEAR, t0-beta y PatchTST-FM con
   escenarios de sus errores, en pérdida pinball del costo del día por nivel τ, todos
   calibrados por nivel y bloque de horas. Escribe ``cuadro_curva_tau_calibrada.csv``.
2. ``figura_curva_tau_calendario.png``: lo mismo, con la información de calendario para
   todos. Escribe ``cuadro_curva_tau_calendario.csv``.
3. ``figura_calendario.png``: error absoluto medio del centro por día de la semana del
   día pronosticado, sobre las horas que tienen los cuatro métodos. Escribe
   ``cuadro_error_centro_por_dia.csv``. Aquí los festivos cuentan por su día de la
   semana, así que la última barra son sólo domingos.

El costo del día es lo que pagaría quien compra un megawatt-hora en cada una de las 24
horas: la suma de los 24 precios.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.calibra_escenarios import DETALLE, NIVELES  # noqa: E402
from experiments.curva_tau import compara  # noqa: E402
from experiments.idioma import carpeta, t  # noqa: E402
from experiments.guardar import guarda  # noqa: E402
from experiments.protocol import RESULTS  # noqa: E402

FIGURAS = carpeta(RESULTS.parent / "figuras")
PUBLICACION = RESULTS / "publicacion"
VERDE = "#1b5e20"
MAGENTA = "#d81b60"
SIN_CALENDARIO = [("LEAR", "LEAR"),
                  ("t0-beta+errores", t("t0-beta, de sus errores",
                                        "t0-beta, from its errors")),
                  ("PatchTST-FM+errores", t("PatchTST-FM, de sus errores",
                                            "PatchTST-FM, from its errors"))]
CON_CALENDARIO = [("LEAR", "LEAR"),
                  ("t0-beta+errores-cal7", t("t0-beta, errores con calendario",
                                             "t0-beta, errors with calendar")),
                  ("PatchTST-FM+errores-cal7", t("PatchTST-FM, errores con calendario",
                                                 "PatchTST-FM, errors with calendar"))]
DIAS = ["lunes", "martes a viernes", "sábado", "domingo"]


def por_dia(partes):
    """Puntajes por día calibrados por nivel y bloque, de los archivos y métodos dados."""
    t = pd.concat(
        pd.read_parquet(str(DETALLE).replace(".parquet", f"{sufijo}.parquet"))
        .query("metodo in @metodos")
        for sufijo, metodos in partes)
    return t[t.variante == "nivel_bloque"]


def curva(t, principal, rivales):
    """El método principal contra cada rival en los diecinueve niveles."""
    filas = []
    for nivel in NIVELES:
        for rival, _ in rivales:
            fila = compara(t, principal, rival, f"pin_{nivel:.2f}")
            fila["tau"] = nivel
            filas.append(fila)
    return pd.DataFrame(filas)


def curvas():
    c1 = curva(por_dia([("", ["Analog-mezcla"] + [r for r, _ in SIN_CALENDARIO])]),
               "Analog-mezcla", SIN_CALENDARIO)
    con = por_dia([("", ["LEAR"]), ("_calendario", ["Analog-mezcla-cal7"]),
                   ("_err_cal", [r for r, _ in CON_CALENDARIO[1:]]),
                   ("_combinados_cal", ["Centro-promedio-cal"])])
    c2 = curva(con, "Analog-mezcla-cal7", CON_CALENDARIO)
    ## el campeón: la combinación que promedia los centros de t0-beta y del análogo
    c3 = curva(con, "Centro-promedio-cal", CON_CALENDARIO)
    return c1, c2, c3


def error_del_centro():
    """Error absoluto medio de la mediana por día de la semana del día pronosticado."""
    def medianas(corrida, metodo):
        partes = []
        for f in sorted((RESULTS / corrida).glob("*.parquet")):
            t = pd.read_parquet(f, columns=["zone", "origin", "method", "step", "q0.5",
                                            "observed"])
            partes.append(t[t.method == metodo])
        return pd.concat(partes).set_index(["zone", "origin", "step"])
    P = {"Analog-mix": medianas("picos_prueba", "Analog-mezcla"),
         "Analog-mix con calendario": medianas("calendario", "Analog-mezcla-cal7"),
         "LEAR": medianas("lear_prueba", "LEAR"),
         "t0-beta": medianas("fundacion_t0-beta", "t0-beta")}
    comun = None
    for v in P.values():
        comun = v.index if comun is None else comun.intersection(v.index)
    D = pd.DataFrame({k: v.loc[comun]["q0.5"] for k, v in P.items()}).reset_index()
    D["y"] = P["LEAR"].loc[comun]["observed"].to_numpy()
    ## el origen es la hora 23 del día anterior al pronosticado
    dia = (pd.to_datetime(D.origin) + pd.Timedelta(hours=1)).dt.dayofweek
    tipo = dia.map(lambda d: DIAS[0] if d == 0 else DIAS[2] if d == 5 else
                   DIAS[3] if d == 6 else DIAS[1])
    tab = pd.DataFrame({k: (D[k] - D.y).abs().groupby(tipo).mean() for k in P}).loc[DIAS]
    tab.loc["todos"] = {k: (D[k] - D.y).abs().mean() for k in P}
    return tab, len(comun)


def figura_curva(series, rivales, sujeto, titulo, nombre):
    """Una curva por cada (cuadro, etiqueta, color) de ``series``, contra cada rival."""
    fig, ejes = plt.subplots(1, len(rivales), figsize=(13, 4.2), sharey=True)
    for eje, (rival, etiqueta) in zip(ejes, rivales):
        for c, nombre_serie, color in series:
            g = c[c.b == rival].sort_values("tau")
            eje.plot(g.tau, g.pct, "o-", ms=3, color=color, label=nombre_serie)
            eje.fill_between(g.tau, g.pct_bajo, g.pct_alto, color=color, alpha=0.15)
        eje.axhline(0, color="black", lw=0.8)
        eje.set_title(t(f"contra {etiqueta}", f"against {etiqueta}"), fontsize=10.5)
        eje.set_xlabel(t("τ (parte de la distribución del costo del día)",
                         "τ (part of the distribution of the daily cost)"))
        eje.grid(alpha=0.3)
    ejes[0].set_ylabel(t(f"{sujeto} menos rival,\n% del rival (abajo de cero gana el "
                         f"{'análogo' if len(series) == 1 else 'método'})",
                         f"{sujeto} minus rival,\n% of the rival (below zero the "
                         f"{'analog method' if len(series) == 1 else 'method'} wins)"))
    if len(series) > 1:
        ejes[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    ruta = FIGURAS / nombre
    guarda(fig, ruta, dpi=160)
    plt.close(fig)
    return ruta


def figura_calendario(tab):
    d = tab.loc[DIAS]
    colores = {"Analog-mix": "#9e9e9e", "Analog-mix con calendario": VERDE,
               "LEAR": "#ef6c00", "t0-beta": "#5e35b1"}
    ## los días y los métodos son las llaves del cuadro que se guarda en CSV: se traducen
    ## sólo al dibujarlos, para que el cuadro salga igual en los dos idiomas
    en = {"Analog-mix": "Analog-mix", "Analog-mix con calendario": "Analog-mix with calendar",
          "LEAR": "LEAR", "t0-beta": "t0-beta", "lunes": "Monday",
          "martes a viernes": "Tuesday to Friday", "sábado": "Saturday",
          "domingo": "Sunday"}
    fig, eje = plt.subplots(figsize=(10, 4.6))
    x, ancho = np.arange(len(d)), 0.2
    for i, metodo in enumerate(d.columns):
        posiciones = x + (i - 1.5) * ancho
        eje.bar(posiciones, d[metodo], ancho, label=t(metodo, en[metodo]),
                color=colores[metodo])
        for xi, v in zip(posiciones, d[metodo]):
            eje.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=7.5)
    eje.set_xticks(x)
    eje.set_xticklabels([t(dia, en[dia]) for dia in d.index])
    eje.set_ylabel(t("error absoluto medio del centro,\npesos por MWh",
                     "mean absolute error of the center,\npesos per MWh"))
    eje.legend(fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    eje.grid(alpha=0.3, axis="y")
    eje.set_ylim(0, 240)
    fig.tight_layout()
    ruta = FIGURAS / "figura_calendario.png"
    guarda(fig, ruta, dpi=160)
    plt.close(fig)
    return ruta


def main():
    c1, c2, c3 = curvas()
    c1.round(4).to_csv(PUBLICACION / "cuadro_curva_tau_calibrada.csv", index=False)
    c2.round(4).to_csv(PUBLICACION / "cuadro_curva_tau_calendario.csv", index=False)
    c3.round(4).to_csv(PUBLICACION / "cuadro_curva_tau_campeon.csv", index=False)
    tab, horas = error_del_centro()
    tab.round(1).to_csv(PUBLICACION / "cuadro_error_centro_por_dia.csv",
                        index_label="dia")
    print(f"error del centro sobre {horas:,d} horas comunes:")
    print(tab.round(1).to_string())
    for ruta in (
            figura_curva([(c1, "Analog-mix", VERDE)], SIN_CALENDARIO, "Analog-mix",
                         t("Escenarios calibrados por nivel y bloque de horas, todos los "
                           "métodos igual", "Scenarios calibrated by level and hour block, "
                           "all methods the same way"), "figura_curva_tau_calibrada.png"),
            figura_curva([(c2, t("Analog-mix con calendario", "Analog-mix with calendar"),
                           VERDE),
                          (c3, t("t0-beta + análogo (campeón)",
                                 "t0-beta + analog (champion)"), MAGENTA)],
                         CON_CALENDARIO, t("Método", "Method"),
                         t("Todos calibrados por nivel y bloque de horas, y con la misma "
                           "información de calendario", "All calibrated by level and hour "
                           "block, and with the same calendar information"),
                         "figura_curva_tau_calendario.png"),
            figura_calendario(tab)):
        print("escrito", ruta)


if __name__ == "__main__":
    main()
