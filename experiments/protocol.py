"""
El protocolo del experimento: tramos, mallas y la configuración congelada.

La serie se parte en tramos que nunca se mezclan hacia atrás:

    banco de búsqueda   los dos años previos a cada origen, rodando
    selección           2020-04-01 a 2021-03-31
    prueba principal    2021-04-01 a 2026-02-28
    ya inspeccionado    2026-03-01 en adelante

El último tramo existe porque sobre él ya corrimos una versión anterior del
experimento: se reporta aparte y no se le llama intocado.

El tramo de selección sí forma parte del banco donde el de prueba busca análogos,
como lo formaría en operación. Lo que no ocurre es lo contrario: nada posterior al
inicio de la prueba entra en la selección.

La configuración no se escribe a mano en ningún módulo. La elige ``select_config``
sobre el tramo de selección y queda congelada en ``results/selected_config.json``;
quien corra el experimento sin ese archivo obtiene un error, no un valor por
omisión.
"""

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"

HORIZON = 24
ORIGIN_STEP_DAYS = 1
SEARCH_YEARS = 2

SELECTION_START = "2020-04-01"
SELECTION_END = "2021-03-31"
TEST_START = "2021-04-01"
TEST_END = "2026-02-28"
INSPECTED_START = "2026-03-01"
INSPECTED_END = "2026-09-30"

## Malla del método. Los años de búsqueda no entran: cambian el conjunto de
## información y no la forma del método, y con ellos la malla dejaría de ser
## comparable contra comparativos que ven entre treinta y ciento veinte días.
WINDOW_GRID = (24, 48, 72, 168)
K_GRID = (10, 20, 40, 100)
SEPARATION_GRID = (0.5, 1.0)

## HEREDADOS, NO ELEGIDOS. Estos tres valores vienen del experimento anterior y el
## barrido de este rediseño los dejó fijos: sólo recorrió gamma. La malla de arriba
## se corrió después, en `grid_sweep.py`, para comprobar y no para elegir: sobre el
## tramo de selección 282 de sus 352 celdas quedan dentro de un error estándar de la
## mejor, y la heredada es una de ellas, así que la malla no la distingue. Estos
## valores se leen de aquí y no se escriben a mano en los módulos, para que su origen
## se vea en el protocolo en vez de esconderse en un script de barrido.
INHERITED_WINDOW = 48
INHERITED_K = 40
INHERITED_SEPARATION = 0.5
GAMMA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

## Malla de los comparativos. El recorte de salida y los pesos por similitud no se
## ajustan: son decisiones de diseño y se reportan como ablación.
BENCH_HISTORY_GRID = (30, 60, 120)
LGBM_GRID = (
    {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300},
    {"num_leaves": 31, "learning_rate": 0.10, "n_estimators": 600},
    {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 300},
    {"num_leaves": 63, "learning_rate": 0.10, "n_estimators": 600},
)
LGBM_TRAIN_YEARS = 3
LGBM_REFIT_DAYS = 30

## Agrupamiento de zonas: se agrupa sobre lo que queda al quitar el factor común,
## porque sobre el precio crudo manda ese factor y se elegiría por nivel de precio
## en vez de por comportamiento.
ZONE_CLUSTER_THRESHOLD = 0.85
ZONE_MANDATORY_EXTREMES = ("spike_rate", "congestion_share", "negative_rate")

## Niveles comunes a todos los métodos. La comparación en crudo usa los nueve que
## todos emiten de fábrica; la calibrada usa la rejilla completa, porque la capa de
## calibración pone a todos en los mismos niveles.
COMMON_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FULL_LEVELS = (0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975)

SELECTED_FILE = RESULTS / "selected_config.json"
SELECTED_BENCH_FILE = RESULTS / "selected_benchmarks.json"
SELECTED_ZONES_FILE = RESULTS / "selected_zones.json"


def build_origins(start: str, end: str, series: pd.Series) -> pd.DatetimeIndex:
    """
    Orígenes a las 23:00, uno por día, con horizonte completo observado.

    La hora del índice es la última observada, no la hora en que se emite el
    pronóstico. Con fechas concretas, para que no quede ambiguo:

        6 de octubre, 17:00   el operador publica los precios de TODO el 7 de octubre
        7 de octubre, 10:00   cierra la ventana para ofertar por el 8 de octubre
        origen                7 de octubre, 23:00 — la última hora ya publicada
        horizonte             8 de octubre, de 00:00 a 23:00

    De modo que la decisión se toma la mañana del **mismo día** del origen, antes de
    las 10:00, y las horas del origen ya se conocen porque se publicaron la tarde
    anterior. Una versión previa de esta nota decía «la mañana del día siguiente», que
    está corrido un día y hace parecer que el conjunto de información es otro.

    Nada de esto cambia la forma del problema: la serie está completa hasta el origen
    y se extiende 24 horas contiguas. Lo que sí vale declarar en el artículo es que
    entre el momento de decidir y la última hora pronosticada pasan 38 horas, no 24.
    """
    days = pd.date_range(start, end, freq=f"{ORIGIN_STEP_DAYS}D")
    origins = pd.DatetimeIndex([day.replace(hour=23) for day in days])
    return origins[origins + pd.Timedelta(hours=HORIZON) <= series.index.max()]


def _load(path: Path, script: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"falta {path.name}: corre experiments/{script} antes del experimento")
    with open(path) as handle:
        return json.load(handle)


def load_selected() -> dict:
    """Configuración del método elegida en el tramo de selección."""
    return _load(SELECTED_FILE, "select_config.py")


def load_selected_benchmarks() -> dict:
    """Configuración de los comparativos elegida en el tramo de selección."""
    return _load(SELECTED_BENCH_FILE, "selection_benchmarks.py")


def load_selected_zones() -> dict:
    """Panel de zonas elegido en el tramo de selección."""
    return _load(SELECTED_ZONES_FILE, "select_zones.py")
