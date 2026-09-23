"""
Recalibración del ensamble con los errores de orígenes pasados.

Los miembros se eligieron por parecerse al presente, así que se parecen entre
sí y el ensamble sale estrecho: promete más cobertura de la que entrega. El
error estandarizado de los últimos orígenes ya observados devuelve esa
cobertura, y el precio que se paga es el ancho.

Escribe results/calibration_raw.csv y su resumen.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_probabilistic.analog_probabilistic import (  # noqa: E402
    build_ensemble, calibrate_quantiles, ensemble_quantiles)
from experiments.backtest import (HORIZON, K_ANALOGS, RESULTS,  # noqa: E402
                                  SEARCH_YEARS, SEPARATION, WINDOW)
from experiments.scores import LEVELS, evaluate  # noqa: E402
from experiments.series import load_series  # noqa: E402

PAST_ORIGINS = 20
MIN_SPREAD = 1e-6


def main() -> None:
    series = load_series("pml_mda")
    origins = pd.DatetimeIndex(sorted(
        pd.read_csv(RESULTS / "backtest_raw.csv",
                    parse_dates=["origin"]).origin.unique()))

    standardized: list[np.ndarray] = []
    rows = []
    for origin in origins:
        history = series.loc[:origin].values[-SEARCH_YEARS * 8760:]
        observed = series.loc[origin + pd.Timedelta(hours=1):
                              origin + pd.Timedelta(hours=HORIZON)].values
        if len(observed) != HORIZON:
            continue

        members, _ = build_ensemble(history, window=WINDOW, horizon=HORIZON,
                                    k=K_ANALOGS, separation=SEPARATION)
        quantiles = ensemble_quantiles(members, LEVELS)
        median = quantiles[int(np.argmin(np.abs(LEVELS - 0.5)))]
        spread = np.maximum(
            (np.percentile(members, 90, axis=0)
             - np.percentile(members, 10, axis=0)) / 2.0, MIN_SPREAD)

        if len(standardized) >= PAST_ORIGINS:
            pool = np.concatenate(standardized[-PAST_ORIGINS:])
            calibrated = calibrate_quantiles(median, spread, pool, LEVELS)
            rows.append({"origin": origin, "variant": "calibrated",
                         **evaluate(observed, calibrated)})
            rows.append({"origin": origin, "variant": "raw",
                         **evaluate(observed, quantiles)})

        standardized.append((observed - median) / spread)

    scores = pd.DataFrame(rows)
    scores.to_csv(RESULTS / "calibration_raw.csv", index=False)
    scores.groupby("variant").mean(numeric_only=True).to_csv(
        RESULTS / "calibration_summary.csv")
    print(scores.groupby("variant")[["crps", "reliability", "sharp80",
                                     "sharp95", "cover80", "cover95",
                                     "winkler95", "mae", "pinball95"]]
          .mean().to_string())
    print("orígenes evaluados:", scores.origin.nunique())


if __name__ == "__main__":
    main()
