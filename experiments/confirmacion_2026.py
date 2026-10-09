"""
Confirmación posterior al tramo de prueba, y prueba prospectiva.

Declarado el 2026-10-09, antes de correr ningún pronóstico de estos tramos.

Tramo 1, posterior. Los días pronosticados de marzo a septiembre de 2026
(``INSPECTED_START`` a ``INSPECTED_END`` en ``protocol.py``). Es posterior al tramo de
prueba y no se usó para ninguna decisión de esta versión: el filtro de calendario, la
calibración por nivel y bloque y la combinación se diseñaron con el tramo de prueba. No
es intacto: una versión anterior del experimento, sin esas tres piezas, corrió sobre él
antes del rediseño del 2026-10-06. Se reporta así, con esa salvedad.

Tramo 2, prospectivo. Los días pronosticados a partir del 2026-10-11, que hoy no
existen. Se evalúa cuando haya al menos 90 días, con el código congelado en el commit
que agrega este archivo y las mismas hipótesis.

Métodos, sin cambiar nada de lo que se usó en el tramo de prueba:
- Analog-mezcla-cal7 y su testigo sin filtro (``run_calendario.py``).
- t0-beta y PatchTST-FM (``run_fundacion.py``).
- LEAR con reajuste semanal (``run_bench.py --refit 7``), como en el tramo de prueba.
- Escenarios de errores con calendario para t0-beta, PatchTST-FM y LEAR: la mediana más
  los 60 errores pasados más recientes del mismo tipo de día. LEAR sin calendario: la
  mediana más sus 60 errores pasados más recientes, que es la receta con que LEAR arma
  sus propios escenarios.
- La combinación que promedia los centros de t0-beta y del análogo
  (``run_combinados_calendario.combina``).
Los errores pasados y la calibración necesitan los días previos, así que cada método se
arma sobre la serie continua del tramo de prueba más el nuevo, y sólo se califican los
días del tramo nuevo. Todos se calibran por nivel y bloque de horas con los 100 días
previos (``calibra_escenarios.py``).

Medida: CRPS ponderada del costo del día, con pesos uniforme y de cola alta, y prueba de
Diebold y Mariano con varianza de Newey y West sobre las diferencias sumadas por día.

Hipótesis, en la cola alta; la distribución completa se reporta junto:
H14. Analog-mezcla-cal7 es mejor que LEAR.
H15. La combinación que promedia los centros es mejor que t0-beta+errores-cal7.
H16. Analog-mezcla-cal7 y t0-beta+errores-cal7 son equivalentes con un margen de 3 por
     ciento: el intervalo de confianza del 90 por ciento de la diferencia queda dentro de
     [-3, +3] (dos pruebas unilaterales al 5 por ciento). Si no queda dentro, no se
     afirma equivalencia.
H17. Analog-mezcla-cal7 es mejor que el mismo análogo sin filtro de calendario.
Se reportan todas, se cumplan o no.

Pasos: ``une`` arma las series continuas, ``errores`` los escenarios de errores,
``combina`` la combinación; luego ``calibra_escenarios.py solo conf`` calibra, y
``analiza`` califica el tramo nuevo.
"""

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_probabilistic.analog_probabilistic import ensemble_quantiles  # noqa: E402
from experiments.forecast_store import (ForecastStore, member_columns,  # noqa: E402
                                        member_matrix)
from experiments.protocol import INSPECTED_START, RESULTS  # noqa: E402
from experiments.run_calendario import tipo_de_dia  # noqa: E402
from experiments.run_combinados_calendario import combina  # noqa: E402
from experiments.scores import LEVELS  # noqa: E402

VENTANA_ERRORES = 60
MARGEN_EQUIVALENCIA = 3.0
INICIO_PROSPECTIVO = "2026-10-11"

## (corrida del tramo de prueba, corrida del tramo nuevo, método en cada una, nombre)
ANALOGOS = [("calendario", "insp_calendario", "Analog-mezcla-cal7", "Analog-mezcla-cal7",
             "Analog-mezcla-cal7"),
            ("picos_prueba", "insp_calendario", "Analog-mezcla", "Analog-mezcla-sinfiltro",
             "Analog-mezcla")]
CENTROS = [("fundacion_t0-beta", "insp_fundacion_t0-beta", "t0-beta"),
           ("fundacion_PatchTST-FM", "insp_fundacion_PatchTST-FM", "PatchTST-FM"),
           ("lear_prueba", "insp_lear", "LEAR")]


def _lee(corrida, zona, metodo, miembros=False):
    carpeta = RESULTS / (f"{corrida}_members" if miembros else corrida)
    t = pd.read_parquet(carpeta / f"{zona}.parquet")
    return t[t.method == metodo]


def _zonas():
    return sorted(p.stem for p in (RESULTS / "insp_calendario").glob("*.parquet"))


def une():
    """Las corridas del análogo, continuas del tramo de prueba al nuevo."""
    for destino in ("conf_analogo", "conf_analogo_members"):
        (RESULTS / destino).mkdir(exist_ok=True)
    for zona in _zonas():
        cuantiles, miembros = [], []
        for prueba, nueva, m_prueba, m_nueva, nombre in ANALOGOS:
            for corrida, metodo, desde_nuevo in ((prueba, m_prueba, False),
                                                 (nueva, m_nueva, True)):
                for lista, es_miembro in ((cuantiles, False), (miembros, True)):
                    t = _lee(corrida, zona, metodo, es_miembro)
                    ## del tramo de prueba sólo lo anterior al nuevo, para no duplicar
                    corte = pd.Timestamp(INSPECTED_START)
                    t = t[t.origin >= corte] if desde_nuevo else t[t.origin < corte]
                    lista.append(t.assign(method=nombre))
        pd.concat(cuantiles).to_parquet(RESULTS / "conf_analogo" / f"{zona}.parquet",
                                        index=False)
        pd.concat(miembros).to_parquet(RESULTS / "conf_analogo_members" / f"{zona}.parquet",
                                       index=False)
        print(f"  unido {zona}", flush=True)


def errores():
    """Escenarios de errores sobre la mediana continua de t0-beta, PatchTST-FM y LEAR."""
    almacen = ForecastStore(LEVELS, RESULTS / "conf_errores.parquet")
    corte = pd.Timestamp(INSPECTED_START)
    for zona in _zonas():
        for prueba, nueva, metodo in CENTROS:
            columnas = ["zone", "origin", "method", "step", "q0.5", "observed"]
            a = _lee(prueba, zona, metodo)[columnas]
            b = _lee(nueva, zona, metodo)[columnas]
            t = pd.concat([a[a.origin < corte], b[b.origin >= corte]])
            por_tipo = {k: [] for k in range(7)}
            todos = []
            for origen, p in sorted(t.groupby("origin"), key=lambda kv: kv[0]):
                p = p.sort_values("step")
                observado, mediana = p["observed"].to_numpy(), p["q0.5"].to_numpy()
                if np.isnan(mediana).any() or len(mediana) != 24:
                    continue
                _, siete = tipo_de_dia(pd.DatetimeIndex([origen + pd.Timedelta(hours=13)]))
                propios = por_tipo[int(siete[0])]
                if len(propios) >= 2:
                    miembros = mediana[None, :] + np.array(propios[-VENTANA_ERRORES:])
                    almacen.add(zona, origen, f"{metodo}+errores-cal7",
                                np.quantile(miembros, LEVELS, axis=0), observado,
                                miembros, float("nan"))
                if metodo == "LEAR" and len(todos) >= 2:
                    miembros = mediana[None, :] + np.array(todos[-VENTANA_ERRORES:])
                    almacen.add(zona, origen, "LEAR", np.quantile(miembros, LEVELS, axis=0),
                                observado, miembros, float("nan"))
                propios.append(observado - mediana)
                todos.append(observado - mediana)
        almacen.flush(zona)
        print(f"  errores {zona}", flush=True)


def combinacion():
    """La combinación que promedia los centros, sobre las series continuas."""
    almacen = ForecastStore(LEVELS, RESULTS / "conf_combinado.parquet")
    for zona in _zonas():
        def bloques(corrida, metodo):
            m = _lee(corrida, zona, metodo, miembros=True)
            cols = member_columns(m)
            return {o: member_matrix(b.sort_values("step"), cols)
                    for o, b in m.groupby("origin")}
        a = bloques("conf_analogo", "Analog-mezcla-cal7")
        r = bloques("conf_errores", "t0-beta+errores-cal7")
        obs = _lee("conf_errores", zona, "t0-beta+errores-cal7")
        observados = {o: b.sort_values("step")["observed"].to_numpy()
                      for o, b in obs.groupby("origin")}
        for origen in sorted(set(a) & set(r) & set(observados)):
            A, B, y = a[origen], r[origen], observados[origen]
            if A.shape[1] != 24 or B.shape[1] != 24 or len(B) < 2 or len(y) != 24:
                continue
            _, promedio = combina(A, B, zlib.crc32(f"{zona}|{origen}".encode()))
            almacen.add(zona, origen, "Centro-promedio-cal",
                        ensemble_quantiles(promedio, LEVELS), y, promedio, 0.0)
        almacen.flush(zona)
        print(f"  combinado {zona}", flush=True)


def analiza(desde=INSPECTED_START, hasta=None, etiqueta="posterior"):
    """Las hipótesis H14 a H17 sobre los días pronosticados desde ``desde``."""
    from experiments.calibra_escenarios import DETALLE, NIVELES
    from experiments.curva_tau import PESOS, compara
    t = pd.read_parquet(str(DETALLE).replace(".parquet", "_conf.parquet"))
    t = t[t.variante == "nivel_bloque"].copy()
    dia = pd.to_datetime(t.origen) + pd.Timedelta(hours=1)
    t = t[dia >= pd.Timestamp(desde)]
    if hasta:
        t = t[dia[t.index] <= pd.Timestamp(hasta)]
    p = t[[f"pin_{a:.2f}" for a in NIVELES]].to_numpy()
    for nombre, peso in PESOS.items():
        t[f"qw_{nombre}"] = 2 * (p * peso(NIVELES)).mean(axis=1)
    t["cubre"] = (t.costo_real >= t["cq_0.05"]) & (t.costo_real <= t["cq_0.95"])
    print(f"tramo {etiqueta}: {t.origen.nunique()} días, {t.zona.nunique()} zonas")
    print(t.groupby("metodo").agg(zona_dias=("origen", "size"),
                                  cobertura=("cubre", "mean"),
                                  toda=("qw_uniforme", "mean"),
                                  cola_alta=("qw_cola_alta", "mean")).round(3).to_string())
    filas = []
    for hipotesis, a, b in [("H14", "Analog-mezcla-cal7", "LEAR"),
                            ("H15", "Centro-promedio-cal", "t0-beta+errores-cal7"),
                            ("H16", "Analog-mezcla-cal7", "t0-beta+errores-cal7"),
                            ("H17", "Analog-mezcla-cal7", "Analog-mezcla")]:
        for medida in ("qw_cola_alta", "qw_uniforme"):
            x = compara(t, a, b, medida)
            ## intervalo del 90 por ciento, para la prueba de equivalencia
            error = (x["pct_alto"] - x["pct"]) / 1.96
            x.update(hipotesis=hipotesis, ic90_bajo=x["pct"] - 1.645 * error,
                     ic90_alto=x["pct"] + 1.645 * error)
            filas.append(x)
    r = pd.DataFrame(filas)
    for x in r.itertuples():
        nota = ""
        if x.hipotesis == "H16":
            dentro = -MARGEN_EQUIVALENCIA < x.ic90_bajo and x.ic90_alto < MARGEN_EQUIVALENCIA
            nota = (f"  IC90 [{x.ic90_bajo:+.1f}, {x.ic90_alto:+.1f}] "
                    f"{'dentro' if dentro else 'fuera'} de ±{MARGEN_EQUIVALENCIA:.0f}")
        print(f"   {x.hipotesis} {x.a:22s} vs {x.b:22s} {x.medida[3:]:10s} "
              f"{x.pct:+5.1f}%  t={x.t:+.2f}{nota}")
    r.round(4).to_csv(RESULTS / "publicacion" / f"cuadro_confirmacion_{etiqueta}.csv",
                      index=False)


if __name__ == "__main__":
    paso = sys.argv[1] if len(sys.argv) > 1 else ""
    {"une": une, "errores": errores, "combina": combinacion, "analiza": analiza,
     "prospectivo": lambda: analiza(INICIO_PROSPECTIVO, etiqueta="prospectivo")}[paso]()
