"""
Las figuras de días concretos, con los escenarios ya calibrados por nivel y bloque de horas.

Las figuras anteriores (``figuras_picos.py``) tenían dos problemas que éstas corrigen:

1. Usaban los escenarios CRUDOS, cuando los cuadros ahora comparan escenarios calibrados.
2. La franja de la cuenta del día sumaba los bordes de cada hora. El cuantil de una suma no
   es la suma de los cuantiles, así que eso no era el intervalo de la cuenta. Aquí la cuenta
   se calcula como en los cuadros: se suma cada escenario y se leen los cuantiles de esas
   sumas.

Los días son los mismos de las figuras anteriores, para poder comparar antes y después. Esos
días se eligieron por el resultado (cubiertos por el análogo y no por t0-beta, o cubiertos
por los dos), así que ilustran y no prueban. La figura de días que se anunciaban caros usa
una regla legítima: el decil más caro del último día conocido al decidir.

Uso: python experiments/figuras_calibradas.py
"""

import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.calibra_escenarios import (HISTORIA_MINIMA, TOPE_FACTOR,  # noqa: E402
                                            VENTANA_RODANTE, factores)
from experiments.conformal import escala  # noqa: E402
from experiments.dilema import con_dia_anterior, deciles  # noqa: E402
from experiments.figuras_picos import elige_dias  # noqa: E402
from experiments.forecast_store import member_columns, member_matrix  # noqa: E402
from experiments.protocol import RESULTS  # noqa: E402

FIGURAS = RESULTS.parent / "figuras"
VARIANTE = "nivel_bloque"
FUENTES = {
    "Analog-mezcla": ("picos_prueba", "Analog-mezcla"),
    "LEAR": ("lear_prueba", "LEAR"),
    "t0-beta (campeón)": ("err_t0-beta", "t0-beta+errores"),
}
COLORES = {"Analog-mezcla": "#1b5e20", "LEAR": "#ef6c00", "t0-beta (campeón)": "#5e35b1"}


@lru_cache(maxsize=None)
def calibrados(corrida, metodo, zona, origen, variante=VARIANTE):
    """Los escenarios de un día, calibrados con lo observado en los 100 días previos."""
    m = pd.read_parquet(RESULTS / f"{corrida}_members" / f"{zona}.parquet")
    m = m[m.method == metodo]
    obs = pd.read_parquet(RESULTS / corrida / f"{zona}.parquet",
                          columns=["origin", "step", "observed"]) \
        .drop_duplicates(["origin", "step"])
    observados = {o: b.sort_values("step")["observed"].to_numpy()
                  for o, b in obs.groupby("origin")}
    columnas = member_columns(m)
    H = {k: [] for k in ("obs", "med", "arr", "aba")}
    for o, b in sorted(m.groupby("origin"), key=lambda kv: kv[0]):
        if o > origen:
            break
        y = observados.get(o)
        E = member_matrix(b.sort_values("step"), columnas)
        if y is None or len(y) != 24 or E.shape[1] != 24 or len(E) < 2:
            continue
        med = np.median(E, axis=0)
        q05, q95 = np.quantile(E, [0.05, 0.95], axis=0)
        if o == origen:
            if len(H["obs"]) < HISTORIA_MINIMA:
                return None
            HH = {k: np.array(v[-VENTANA_RODANTE:]) for k, v in H.items()}
            f_a, f_b = factores(variante, HH, med, q95 - med, med - q05, None)
            return {"escenarios": escala(E, med, np.minimum(f_a, TOPE_FACTOR),
                                         np.minimum(f_b, TOPE_FACTOR)),
                    "crudos": E, "real": y}
        H["obs"].append(y)
        H["med"].append(med)
        H["arr"].append(q95 - med)
        H["aba"].append(med - q05)
    return None


def validos(candidatos, cuantos=4):
    """Los primeros días en que los tres métodos tienen historia para calibrarse."""
    dias = []
    for zona, origen in candidatos:
        if all(calibrados(c, m, zona, origen) is not None for c, m in FUENTES.values()):
            dias.append((zona, origen))
        if len(dias) == cuantos:
            break
    return dias


def una_figura(dias, nombre, titulo, nota):
    dias = validos(dias)
    n = len(dias)
    fig, ejes = plt.subplots(2, n, figsize=(4.5 * n, 6.8),
                             gridspec_kw={"height_ratios": [2.4, 1]})
    ejes = ejes.reshape(2, n)
    horas = np.arange(1, 25)
    for col, (zona, origen) in enumerate(dias):
        arriba, abajo = ejes[0, col], ejes[1, col]
        real, cuentas = None, {}
        for etiqueta, (corrida, metodo) in FUENTES.items():
            d = calibrados(corrida, metodo, zona, origen)
            if d is None:
                continue
            E, real = d["escenarios"], d["real"]
            q05, q50, q95 = np.quantile(E, [0.05, 0.5, 0.95], axis=0)
            arriba.fill_between(horas, q05, q95, alpha=0.12, color=COLORES[etiqueta],
                                linewidth=0)
            arriba.plot(horas, q95, color=COLORES[etiqueta], lw=1.8, label=etiqueta)
            arriba.plot(horas, q05, color=COLORES[etiqueta], lw=0.8, ls=":")
            sumas = E.sum(axis=1)
            cuentas[etiqueta] = np.quantile(sumas, [0.05, 0.5, 0.95])
        if real is None:
            continue
        arriba.plot(horas, real, color="black", lw=2.6, label="precio observado")
        arriba.set_title(f"{zona.replace('_', ' ')} · día pronosticado "
                         f"{(pd.Timestamp(origen) + pd.Timedelta(days=1)).date()}",
                         fontsize=9.5)
        arriba.set_xlabel("hora del día")
        arriba.grid(alpha=0.25, lw=0.5)
        cuenta_real = float(real.sum())
        for fila, (etiqueta, (lo, me, hi)) in enumerate(cuentas.items()):
            y = len(cuentas) - fila
            cubre = lo <= cuenta_real <= hi
            abajo.plot([lo, hi], [y, y], color=COLORES[etiqueta],
                       lw=5 if cubre else 2.2, alpha=0.9 if cubre else 0.6,
                       solid_capstyle="butt")
            abajo.plot([me], [y], "o", color=COLORES[etiqueta], ms=5)
            abajo.text(hi, y + 0.28, f"{'cubre' if cubre else 'no cubre'} · "
                       f"ancho {(hi - lo) / 1000:,.0f} mil", fontsize=6.8, ha="right",
                       color="#1b5e20" if cubre else "#b71c1c")
        abajo.axvline(cuenta_real, color="black", lw=2.2)
        abajo.set_yticks(range(1, len(cuentas) + 1))
        abajo.set_yticklabels(list(cuentas)[::-1], fontsize=8)
        abajo.set_ylim(0.4, len(cuentas) + 0.75)
        abajo.xaxis.set_major_locator(MaxNLocator(5))
        abajo.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}"))
        abajo.set_xlabel("cuenta del día, miles de pesos (suma de los 24 precios)")
        abajo.grid(alpha=0.25, lw=0.5, axis="x")
        abajo.set_title(f"cuenta real: {cuenta_real:,.0f}", fontsize=9)
    ejes[0, 0].set_ylabel("precio, pesos por MWh")
    ejes[0, 0].legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle(titulo, fontsize=12.5, y=0.995)
    fig.text(0.5, -0.04, nota, ha="center", fontsize=8.3, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    FIGURAS.mkdir(exist_ok=True)
    ruta = FIGURAS / nombre
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return ruta


def dias_normales():
    """Los mismos días de la figura anterior de días normales (deciles 4 y 5)."""
    d = pd.read_parquet(RESULTS / "costo_diario_por_dia.parquet")
    d["decil"] = deciles(d.costo_real)
    medio = d[d.decil.isin([4, 5])]
    piv = medio.pivot_table(index=["zona", "origen"], columns="metodo",
                            values="dentro_90")
    ambos = piv[(piv["Analog-mezcla"] == 1) & (piv["t0-beta+errores"] == 1)]
    vistos, dias = set(), []
    for (zona, origen) in ambos.index:
        mes = pd.Timestamp(origen).strftime("%Y-%m")
        if mes in vistos or zona in vistos:
            continue
        vistos.update({mes, zona})
        dias.append((zona, origen))
        if len(dias) == 20:
            break
    return dias


def dias_anunciados_caros():
    """Regla legítima: decil más caro del último día conocido al decidir."""
    d = pd.read_parquet(RESULTS / "costo_diario_por_dia.parquet",
                        columns=["metodo", "zona", "origen", "costo_real"])
    d = con_dia_anterior(d[d.metodo == "t0-beta+errores"])
    d["dk"] = deciles(d.costo_conocido)
    alto = d[d.dk == 10].sort_values("costo_conocido", ascending=False)
    vistos, dias = set(), []
    for r in alto.itertuples():
        mes = pd.Timestamp(r.origen).strftime("%Y-%m")
        if mes in vistos or r.zona in vistos:
            continue
        vistos.update({mes, r.zona})
        dias.append((r.zona, r.origen))
        if len(dias) == 20:
            break
    return dias


def figura_por_hora():
    """El campeón por hora del día: ancho y cobertura, sin calibrar y con cada ajuste."""
    h = pd.read_csv(RESULTS / "calibrados_por_hora.csv")
    h = h.groupby(["metodo", "variante", "hora"])[["cubre", "ancho", "n"]].sum()
    h["cubre"], h["ancho"] = h.cubre / h.n, h.ancho / h.n
    estilos = {"crudo": ("sin calibrar", "#9e9e9e", ":"),
               "proporcional": ("ajuste lineal", "#ef6c00", "--"),
               "nivel": ("por nivel", "#1e88e5", "-."),
               "nivel_bloque": ("por nivel y bloque de horas", "#5e35b1", "-")}
    fig, ejes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    for col, metodo in enumerate(["t0-beta+errores", "Analog-mezcla"]):
        a = h.loc[metodo]
        for v, (nombre, color, ls) in estilos.items():
            x = a.loc[v].index + 1
            ejes[0, col].plot(x, a.loc[v].ancho, color=color, ls=ls, lw=2.2 if
                              v == "nivel_bloque" else 1.5, label=nombre)
            ejes[1, col].plot(x, a.loc[v].cubre, color=color, ls=ls, lw=2.2 if
                              v == "nivel_bloque" else 1.5, label=nombre)
        ejes[1, col].axhline(0.90, color="black", lw=1)
        ejes[1, col].text(1, 0.903, "lo prometido: 0.90", fontsize=8)
        titulo = "t0-beta con escenarios de sus errores (campeón)" \
            if col == 0 else "Analog-mezcla"
        ejes[0, col].set_title(titulo, fontsize=11)
        ejes[1, col].set_xlabel("hora del día")
        for e in ejes[:, col]:
            e.grid(alpha=0.25, lw=0.5)
            for x0, x1 in [(1, 6), (19, 22)]:
                e.axvspan(x0, x1, color="#fff3e0" if x0 > 10 else "#e3f2fd", alpha=0.6,
                          lw=0)
    ejes[0, 0].set_ylabel("ancho medio del intervalo de 90%,\npesos por MWh")
    ejes[1, 0].set_ylabel("cobertura por hora")
    ejes[0, 0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, 1.0))
    for e in ejes[0]:
        e.text(1.3, e.get_ylim()[0] + 0.04 * (e.get_ylim()[1] - e.get_ylim()[0]),
               "madrugada", fontsize=8, color="#1565c0")
    for e in ejes[0]:
        e.text(19.2, e.get_ylim()[0] + 0.04 * (e.get_ylim()[1] - e.get_ylim()[0]),
               "pico de la tarde", fontsize=8, color="#e65100")
    fig.suptitle("Cubrir los picos sin exagerar la madrugada: ancho y cobertura por hora, "
                 "25 zonas, abril 2021 a febrero 2026", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ruta = FIGURAS / "figura_ancho_por_hora_ajuste_bloques.png"
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return ruta


def main():
    comun = ("Los tres métodos con sus escenarios calibrados por nivel y bloque de horas, "
             "con lo observado en los 100 días previos. ARRIBA: banda del 90% por hora; "
             "línea gruesa, borde de arriba; punteada, borde de abajo. ABAJO: intervalo "
             "del 90% de la CUENTA DEL DÍA, calculado sumando cada escenario; el punto es "
             "la mediana y la línea negra, la cuenta real. ")
    dias, cumplen, total = elige_dias(20)
    r1 = una_figura(dias, "figura_picos_cubiertos_ajuste_bloques.png",
                    "Días caros, con ajuste por nivel y bloque de horas",
                    comun + "Son los mismos días de la figura anterior, elegidos por el "
                    "resultado con los escenarios crudos (el análogo cubría y t0-beta no): "
                    "ilustran, no prueban.")
    r2 = una_figura(dias_normales(), "figura_dias_normales_ajuste_bloques.png",
                    "Días normales, con ajuste por nivel y bloque de horas",
                    comun + "Son los mismos días de la figura anterior: deciles 4 y 5 de la "
                    "cuenta, donde los dos cubrían con los escenarios crudos.")
    r3 = una_figura(dias_anunciados_caros(), "figura_anunciados_caros_ajuste_bloques.png",
                    "Días que se anunciaban caros: el día conocido al decidir fue de los "
                    "más caros",
                    comun + "Regla legítima, sin mirar el resultado: decil más caro del "
                    "último día publicado al decidir; el más caro de cada zona y mes.")
    r4 = figura_por_hora()
    for r in (r1, r2, r3, r4):
        print("escrito", r)


if __name__ == "__main__":
    main()
