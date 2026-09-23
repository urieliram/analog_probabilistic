"""
El mapa de miembro decide el desempeño.

Las tres pendientes de la familia se corren sobre los mismos orígenes, los
mismos análogos y la misma evaluación, así que la diferencia entre ellas es el
mapa y nada más. Se añaden dos variantes de la pendiente de mínimos cuadrados:
con pesos por similitud y sin recorte de salida.

Escribe results/ablation_raw.csv, su resumen y la tasa de activación del
recorte.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_probabilistic.analog_probabilistic import (  # noqa: E402
    build_ensemble, ensemble_quantiles)
from experiments.backtest import (HORIZON, K_ANALOGS, RESULTS,  # noqa: E402
                                  SEARCH_YEARS, SEPARATION, WINDOW)
from experiments.scores import LEVELS, evaluate  # noqa: E402
from experiments.series import load_series  # noqa: E402

VARIANTS = ("ols", "rescale", "raw", "ols-weighted")


def clipped_fraction(series: np.ndarray, members: np.ndarray) -> float:
    """Fracción de horas-miembro que tocan alguno de los dos topes de salida."""
    low = float(np.percentile(series, 1)) - 2.0 * float(np.std(series))
    high = float(series.max())
    return float(np.mean((members <= low) | (members >= high)))


def variant_quantiles(history: np.ndarray, variant: str, clip: bool = True):
    map_name = "ols" if variant.startswith("ols") else variant
    members, similarities = build_ensemble(
        history, window=WINDOW, horizon=HORIZON, k=K_ANALOGS,
        separation=SEPARATION, map_name=map_name, clip=clip,
    )
    weights = None
    if variant == "ols-weighted":
        weights = similarities / similarities.sum()
    return ensemble_quantiles(members, LEVELS, weights), members


def main() -> None:
    series = load_series("pml_mda")
    origins = pd.DatetimeIndex(sorted(
        pd.read_csv(RESULTS / "backtest_raw.csv",
                    parse_dates=["origin"]).origin.unique()))

    rows, clip_rates = [], []
    for origin in origins:
        history = series.loc[:origin].values[-SEARCH_YEARS * 8760:]
        observed = series.loc[origin + pd.Timedelta(hours=1):
                              origin + pd.Timedelta(hours=HORIZON)].values
        if len(observed) != HORIZON:
            continue

        for variant in VARIANTS:
            started = time.time()
            quantiles, members = variant_quantiles(history, variant)
            rows.append({"origin": origin, "variant": variant,
                         "time_s": time.time() - started,
                         **evaluate(observed, quantiles)})
            if variant == "ols":
                clip_rates.append(clipped_fraction(history, members))

        quantiles, _ = variant_quantiles(history, "ols", clip=False)
        rows.append({"origin": origin, "variant": "ols-noclip",
                     "time_s": np.nan, **evaluate(observed, quantiles)})

    scores = pd.DataFrame(rows)
    scores.to_csv(RESULTS / "ablation_raw.csv", index=False)
    scores.groupby("variant").mean(numeric_only=True).to_csv(
        RESULTS / "ablation_summary.csv")
    pd.Series({"clip_rate_mean": float(np.mean(clip_rates)),
               "clip_rate_max": float(np.max(clip_rates))}).to_csv(
        RESULTS / "clip_rate.csv")

    print(scores.groupby("variant")[["crps", "reliability", "sharp95",
                                     "cover95", "pinball95", "mae"]]
          .mean().to_string())
    print("tasa de recorte media:", np.mean(clip_rates))


if __name__ == "__main__":
    main()
