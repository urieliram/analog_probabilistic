"""
Sensibilidad a la ventana presente, al número de análogos y a la separación.

La malla se fijó antes de correr el experimento principal y no se usó para
elegir la configuración que reporta el artículo. Se evalúa sobre una submuestra
de orígenes, así que sus puntajes no son comparables con los del cuadro
principal: lo que importa es el orden.

Escribe results/sensitivity.csv.
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_probabilistic.analog_probabilistic import (  # noqa: E402
    build_ensemble, ensemble_quantiles)
from experiments.backtest import HORIZON, RESULTS, SEARCH_YEARS  # noqa: E402
from experiments.scores import LEVELS, evaluate  # noqa: E402
from experiments.series import load_series  # noqa: E402

WINDOW_GRID = (24, 48, 168)
K_GRID = (10, 20, 40, 100)
SEPARATION_GRID = (0.5, 1.0)
SUBSAMPLE = 30


def main() -> None:
    series = load_series("pml_mda")
    origins = pd.DatetimeIndex(sorted(
        pd.read_csv(RESULTS / "backtest_raw.csv",
                    parse_dates=["origin"]).origin.unique()))
    origins = origins[::max(1, len(origins) // SUBSAMPLE)][:SUBSAMPLE]

    rows = []
    for window in WINDOW_GRID:
        for k in K_GRID:
            for separation in SEPARATION_GRID:
                scores = []
                for origin in origins:
                    history = series.loc[:origin].values[-SEARCH_YEARS * 8760:]
                    observed = series.loc[
                        origin + pd.Timedelta(hours=1):
                        origin + pd.Timedelta(hours=HORIZON)].values
                    if len(observed) != HORIZON:
                        continue
                    started = time.time()
                    members, _ = build_ensemble(
                        history, window=window, horizon=HORIZON, k=k,
                        separation=separation)
                    quantiles = ensemble_quantiles(members, LEVELS)
                    score = evaluate(observed, quantiles)
                    score["time_s"] = time.time() - started
                    score["k_real"] = len(members)
                    scores.append(score)

                average = pd.DataFrame(scores).mean(numeric_only=True)
                rows.append({"window": window, "k": k, "separation": separation,
                             "n_origins": len(scores), **average.to_dict()})
                print(f"ventana={window:3d} k={k:3d} separacion={separation} "
                      f"crps={average['crps']:7.2f} "
                      f"fiabilidad={average['reliability']:.3f}", flush=True)

    pd.DataFrame(rows).to_csv(RESULTS / "sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
