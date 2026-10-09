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
bloque de horas con los cien días previos, y la cuenta del día calculada sumando cada
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

CON_CALENDARIO = {
    "Analog-mix con calendario": ("calendario", "Analog-mezcla-cal7"),
    "LEAR": ("lear_prueba", "LEAR"),
    "t0-beta con calendario (campeón)": ("err_cal", "t0-beta+errores-cal7"),
}
COLORES_CAL = {"Analog-mix con calendario": "#1b5e20", "LEAR": "#ef6c00",
               "t0-beta con calendario (campeón)": "#5e35b1",
               "Analog-mix sin calendario": "#9e9e9e"}
DOMINGOS = {
    "Analog-mix sin calendario": ("picos_prueba", "Analog-mezcla"),
    "Analog-mix con calendario": ("calendario", "Analog-mezcla-cal7"),
    "t0-beta con calendario (campeón)": ("err_cal", "t0-beta+errores-cal7"),
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


def dias_sorteados(cuantos=4):
    """Días al azar del tramo de prueba, con semilla fija, de zonas y meses distintos."""
    import numpy as np
    from experiments.protocol import load_selected_zones
    zonas = load_selected_zones()["zonas"]
    t = pd.read_parquet(RESULTS / "calendario" / f"{zonas[0]}.parquet", columns=["origin"])
    origenes = sorted(t.origin.unique())[40:]
    rng = np.random.default_rng(SEMILLA_SORTEO)
    vistos, dias = set(), []
    while len(dias) < cuantos:
        zona = zonas[rng.integers(len(zonas))]
        origen = pd.Timestamp(origenes[rng.integers(len(origenes))])
        mes = origen.strftime("%Y-%m")
        if zona in vistos or mes in vistos:
            continue
        vistos.update({zona, mes})
        dias.append((zona, origen))
    return dias


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
            ("t0-beta+errores-cal7", "t0-beta, errores con calendario (campeón)"),
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
             "ARRIBA: banda del 90% por hora; línea gruesa, borde de arriba; punteada, "
             "borde de abajo. ABAJO: intervalo del 90% de la CUENTA DEL DÍA, calculado "
             "sumando cada escenario; el punto es la mediana y la línea negra, la cuenta "
             "real. ")
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
    r3 = figura_por_hora()
    for r in (r1, r2, r3, r4, r5):
        print("escrito", r)


if __name__ == "__main__":
    main()
