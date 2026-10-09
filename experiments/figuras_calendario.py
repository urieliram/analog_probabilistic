"""
Las figuras del artículo con la configuración de calendario: el método análogo busca sólo
continuaciones del mismo tipo de día, y los escenarios de errores de t0-beta usan sólo
errores pasados del mismo tipo de día.

Cinco figuras:
1. ``figura_anunciados_caros_calendario.png``: los mismos cuatro días de la figura de días
   que se anunciaban caros (regla sin mirar el resultado), con los métodos con calendario.
2. ``figura_domingos_calendario.png``: cuatro domingos fijados de antemano, sin mirar el
   resultado (el primer domingo no festivo de marzo, junio, septiembre y diciembre de
   2024, en cuatro zonas de regiones distintas), con el análogo sin y con calendario y
   t0-beta con calendario. Muestra el mecanismo: sin calendario el análogo pronostica un
   domingo con la forma de un día laborable.
3. ``figura_ancho_por_hora_calendario.png``: ancho y cobertura por hora con cada
   calibración, para t0-beta y el análogo, los dos con calendario.
4. ``figura_festivos_calendario.png``: cuatro festivos entre semana escogidos antes de
   mirar el resultado de cada método en ellos (1 de mayo de 2023, 16 de septiembre de
   2024, 25 de diciembre de 2024 y 1 de enero de 2025) en cuatro zonas de regiones
   distintas, con el análogo sin y con calendario y t0-beta con calendario. El filtro
   trata el festivo como domingo; la mediana de t0-beta no sabe que es festivo, aunque
   sus errores sí vienen de domingos y festivos pasados.
5. ``figura_dias_sorteados_calendario.png``: cuatro días sorteados del tramo de prueba
   con semilla fija, de zonas y meses distintos, con los tres métodos con calendario.

Usa las mismas funciones de ``figuras_calibradas.py``: escenarios calibrados por nivel y
bloque de horas con los cien días previos, y el costo del día calculado sumando cada
escenario.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.figuras_calibradas as fc  # noqa: E402
from experiments.protocol import RESULTS  # noqa: E402
from experiments.run_calendario import tipo_de_dia  # noqa: E402

FIGURAS = RESULTS.parent / "figuras"

## el campeón es la combinación que promedia los centros de t0-beta y del análogo, los
## dos con calendario (experiments/run_combinados_calendario.py, hipótesis H13)
CAMPEON = ("combinados_cal", "Centro-promedio-cal")
CON_CALENDARIO = {
    "Analog-mix con calendario": ("calendario", "Analog-mezcla-cal7"),
    "LEAR": ("lear_prueba", "LEAR"),
    "t0-beta con calendario": ("err_cal", "t0-beta+errores-cal7"),
    "t0-beta + análogo (campeón)": CAMPEON,
}
COLORES_CAL = {"Analog-mix con calendario": "#1b5e20", "LEAR": "#ef6c00",
               "t0-beta con calendario": "#5e35b1",
               "t0-beta + análogo (campeón)": "#d81b60",
               "Analog-mix sin calendario": "#9e9e9e"}
DOMINGOS = {
    "Analog-mix sin calendario": ("picos_prueba", "Analog-mezcla"),
    "Analog-mix con calendario": ("calendario", "Analog-mezcla-cal7"),
    "t0-beta con calendario": ("err_cal", "t0-beta+errores-cal7"),
    "t0-beta + análogo (campeón)": CAMPEON,
}
## zonas de regiones distintas, fijadas antes de mirar ningún resultado de estos días
ZONAS_DOMINGO = ["merida", "centro_sur", "chihuahua", "culiacan"]
MESES_DOMINGO = [(2024, 3), (2024, 6), (2024, 9), (2024, 12)]


def _con(fuentes, colores):
    fc.FUENTES = fuentes
    fc.COLORES = colores


def domingos():
    """El primer domingo no festivo de cada mes elegido, uno por zona."""
    dias = []
    for zona, (anio, mes) in zip(ZONAS_DOMINGO, MESES_DOMINGO):
        fechas = pd.date_range(f"{anio}-{mes:02d}-01", periods=14, freq="D")
        _, siete = tipo_de_dia(fechas + pd.Timedelta(hours=12))
        for fecha, tipo in zip(fechas, siete):
            if fecha.dayofweek == 6 and tipo == 6 and fecha.date() not in fc_festivos():
                ## el origen es la hora 23 del día anterior al pronosticado
                dias.append((zona, fecha - pd.Timedelta(hours=1)))
                break
    return dias


FESTIVOS_ELEGIDOS = [("villahermosa", "2023-05-01"), ("hermosillo", "2024-09-16"),
                     ("tampico", "2024-12-25"), ("zamora", "2025-01-01")]
SEMILLA_SORTEO = 20261008


def festivos_elegidos():
    return [(z, pd.Timestamp(f) - pd.Timedelta(hours=1)) for z, f in FESTIVOS_ELEGIDOS]


SEMILLA_REJILLA = 20261009


def dias_sorteados(cuantos=4, semilla=None, excluir=()):
    """Días al azar del tramo de prueba, con semilla fija, de zonas y meses distintos.

    Con 25 zonas, si se piden más días que zonas se permite repetir zona, pero nunca mes.
    """
    import numpy as np
    from experiments.protocol import load_selected_zones
    zonas = load_selected_zones()["zonas"]
    t = pd.read_parquet(RESULTS / "calendario" / f"{zonas[0]}.parquet", columns=["origin"])
    origenes = sorted(t.origin.unique())[40:]
    rng = np.random.default_rng(SEMILLA_SORTEO if semilla is None else semilla)
    zonas_vistas, meses_vistos, dias = set(), set(), []
    excluir = {(z, pd.Timestamp(o)) for z, o in excluir}
    while len(dias) < cuantos:
        zona = zonas[rng.integers(len(zonas))]
        origen = pd.Timestamp(origenes[rng.integers(len(origenes))])
        mes = origen.strftime("%Y-%m")
        if mes in meses_vistos or (zona in zonas_vistas and cuantos <= len(zonas)) \
                or (zona, origen) in excluir:
            continue
        zonas_vistas.add(zona)
        meses_vistos.add(mes)
        dias.append((zona, origen))
    return dias


def figura_rejilla(dias, nombre, titulo, nota):
    """Sólo las series por hora, en una rejilla: mediana y banda del 90% de cada método,
    y cuántas de las 24 horas cayeron dentro de la banda."""
    import numpy as np
    dias = fc.validos(dias, cuantos=16)
    fig, ejes = plt.subplots(4, 4, figsize=(17, 14))
    horas = np.arange(1, 25)
    for eje, (zona, origen) in zip(ejes.ravel(), dias):
        real, textos = None, []
        for etiqueta, (corrida, metodo) in fc.FUENTES.items():
            d = fc.calibrados(corrida, metodo, zona, origen)
            if d is None:
                continue
            E, real = d["escenarios"], d["real"]
            q05, q50, q95 = np.quantile(E, [0.05, 0.5, 0.95], axis=0)
            color = fc.COLORES[etiqueta]
            eje.fill_between(horas, q05, q95, color=color, alpha=0.10, linewidth=0)
            eje.plot(horas, q95, color=color, lw=0.6)
            eje.plot(horas, q50, color=color, lw=1.8, label=etiqueta)
            dentro = int(((real >= q05) & (real <= q95)).sum())
            textos.append((f"{dentro}/24", color))
        eje.plot(horas, real, color="black", lw=2.2, label="precio observado")
        dia = (pd.Timestamp(origen) + pd.Timedelta(days=1))
        eje.set_title(f"{zona.replace('_', ' ')} · {dia.date()} ({dia.day_name()[:3]})",
                      fontsize=9.5)
        eje.grid(alpha=0.25, lw=0.5)
        eje.tick_params(labelsize=8)
        for i, (texto, color) in enumerate(textos):
            eje.text(0.02 + 0.13 * i, 0.97, texto, transform=eje.transAxes, fontsize=8,
                     color=color, va="top", fontweight="bold")
    for eje in ejes[-1]:
        eje.set_xlabel("hora del día", fontsize=9)
    for eje in ejes[:, 0]:
        eje.set_ylabel("pesos por MWh", fontsize=9)
    manejadores, etiquetas = ejes[0, 0].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc="upper center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(titulo, fontsize=13, y=1.0)
    fig.text(0.5, 0.005, nota, ha="center", fontsize=9, wrap=True)
    fig.tight_layout(rect=[0, 0.025, 1, 0.965])
    ruta = FIGURAS / nombre
    fig.savefig(ruta, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return ruta


def fc_festivos():
    from experiments.run_calendario import FESTIVOS
    return FESTIVOS


def figura_por_hora():
    """Ancho y cobertura por hora con cada calibración, con calendario."""
    h = pd.concat([pd.read_csv(RESULTS / "calibrados_por_hora_calendario.csv"),
                   pd.read_csv(RESULTS / "calibrados_por_hora_err_cal.csv")])
    h = h.groupby(["metodo", "variante", "hora"])[["cubre", "ancho", "n"]].sum()
    h["cubre"], h["ancho"] = h.cubre / h.n, h.ancho / h.n
    estilos = {"crudo": ("sin calibrar", "#9e9e9e", ":"),
               "proporcional": ("ajuste lineal", "#ef6c00", "--"),
               "nivel": ("por nivel", "#1e88e5", "-."),
               "nivel_bloque": ("por nivel y bloque de horas", "#5e35b1", "-")}
    fig, ejes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    for col, (metodo, titulo) in enumerate([
            ("t0-beta+errores-cal7", "t0-beta, errores con calendario"),
            ("Analog-mezcla-cal7", "Analog-mix con calendario")]):
        a = h.loc[metodo]
        for v, (nombre, color, ls) in estilos.items():
            x = a.loc[v].index + 1
            ancho = 2.2 if v == "nivel_bloque" else 1.5
            ejes[0, col].plot(x, a.loc[v].ancho, color=color, ls=ls, lw=ancho, label=nombre)
            ejes[1, col].plot(x, a.loc[v].cubre, color=color, ls=ls, lw=ancho, label=nombre)
        ejes[1, col].axhline(0.90, color="black", lw=1)
        ejes[1, col].text(1, 0.903, "lo prometido: 0.90", fontsize=8)
        ejes[0, col].set_title(titulo, fontsize=11)
        ejes[1, col].set_xlabel("hora del día")
        for e in ejes[:, col]:
            e.grid(alpha=0.25, lw=0.5)
            for x0, x1 in [(1, 6), (19, 22)]:
                e.axvspan(x0, x1, color="#fff3e0" if x0 > 10 else "#e3f2fd", alpha=0.6,
                          lw=0)
    for e in ejes[0]:
        lo, hi = e.get_ylim()
        e.text(1.3, lo + 0.04 * (hi - lo), "madrugada", fontsize=8, color="#1565c0")
        e.text(19.2, lo + 0.04 * (hi - lo), "pico de la tarde", fontsize=8, color="#e65100")
    ejes[0, 0].set_ylabel("ancho medio del intervalo de 90%,\npesos por MWh")
    ejes[1, 0].set_ylabel("cobertura por hora")
    ejes[0, 0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, 1.0))
    fig.suptitle("Con calendario: ancho y cobertura por hora con cada calibración, 25 zonas, "
                 "abril 2021 a febrero 2026", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ruta = FIGURAS / "figura_ancho_por_hora_calendario.png"
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return ruta


def main():
    comun = ("Escenarios calibrados por nivel y bloque de horas con los 100 días previos. "
             "ARRIBA: mediana en línea gruesa y banda del 90% por hora (delgada, borde de "
             "arriba; punteada, borde de abajo). ABAJO: intervalo del 90% del COSTO DEL DÍA (comprar 1 MWh "
             "en cada hora), calculado sumando cada escenario; el punto es la mediana y "
             "la línea negra, el costo real. ")
    _con(CON_CALENDARIO, COLORES_CAL)
    r1 = fc.una_figura(fc.dias_anunciados_caros(), "figura_anunciados_caros_calendario.png",
                       "Días que se anunciaban caros, con calendario para todos",
                       comun + "Regla sin mirar el resultado: decil más caro del último día "
                       "publicado al decidir; el más caro de cada zona y mes. Son los "
                       "mismos días de la figura sin calendario.")
    fc.calibrados.cache_clear()
    _con(DOMINGOS, COLORES_CAL)
    r2 = fc.una_figura(domingos(), "figura_domingos_calendario.png",
                       "Cuatro domingos: el análogo sin y con calendario",
                       comun + "Días fijados antes de mirar el resultado: el primer domingo no "
                       "festivo de marzo, junio, septiembre y diciembre de 2024, en cuatro "
                       "zonas de regiones distintas.")
    fc.calibrados.cache_clear()
    _con(DOMINGOS, COLORES_CAL)
    r4 = fc.una_figura(festivos_elegidos(), "figura_festivos_calendario.png",
                       "Cuatro festivos entre semana: el análogo sin y con calendario",
                       comun + "Festivos escogidos antes de mirar el resultado: 1 de mayo de "
                       "2023, 16 de septiembre de 2024, 25 de diciembre de 2024 y 1 de enero "
                       "de 2025, en cuatro zonas de regiones distintas. El filtro trata el "
                       "festivo como domingo; la mediana de t0-beta no sabe que es festivo, "
                       "aunque sus errores sí vienen de domingos y festivos pasados.")
    fc.calibrados.cache_clear()
    _con(CON_CALENDARIO, COLORES_CAL)
    r5 = fc.una_figura(dias_sorteados(), "figura_dias_sorteados_calendario.png",
                       "Cuatro días cualquiera, sorteados, con calendario para todos",
                       comun + f"Días sorteados del tramo de prueba con semilla fija "
                       f"({SEMILLA_SORTEO}), de zonas y meses distintos.")
    fc.calibrados.cache_clear()
    _con(CON_CALENDARIO, COLORES_CAL)
    r6 = figura_rejilla(
        dias_sorteados(24, semilla=SEMILLA_REJILLA, excluir=dias_sorteados()),
        "figura_rejilla_dias_sorteados.png",
        "Dieciséis días cualquiera, sorteados, con calendario para todos",
        "Escenarios calibrados por nivel y bloque de horas con los 100 días previos. Línea "
        "gruesa: mediana de cada método; sombra: banda del 90% por hora; línea negra: "
        "precio observado. Arriba a la izquierda, cuántas de las 24 horas cayeron dentro "
        f"de la banda de cada método. Días sorteados con semilla fija ({SEMILLA_REJILLA}), "
        "de meses distintos, distintos de los de la galería de cuatro días.")
    r3 = figura_por_hora()
    for r in (r1, r2, r3, r4, r5, r6):
        print("escrito", r)


if __name__ == "__main__":
    main()
