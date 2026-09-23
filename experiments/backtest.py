"""
Comparación del análogo probabilístico contra cuatro comparativos.

Origen rodante cada dos días sobre el precio marginal local del día en
adelanto, horizonte de veinticuatro horas. Cada método recibe la serie hasta el
origen y devuelve los mismos diecinueve cuantiles, de modo que los puntajes son
comparables y la prueba de Diebold-Mariano se calcula sobre los mismos días.

Escribe results/backtest_raw.csv, el resumen por método y el abanico de un día
para la figura del artículo.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_probabilistic.analog_probabilistic import (  # noqa: E402
    build_ensemble, ensemble_quantiles)
from experiments.scores import LEVELS, evaluate  # noqa: E402
from experiments.series import load_series  # noqa: E402

warnings.filterwarnings("ignore")

RESULTS = Path(__file__).resolve().parents[1] / "results"

HORIZON = 24
TEST_START = "2026-03-01"
TEST_END = "2026-08-31"
ORIGIN_STEP_DAYS = 2
SEARCH_YEARS = 2
WINDOW = 48
K_ANALOGS = 40
SEPARATION = 0.5
BENCH_HISTORY_DAYS = 60
LGBM_TRAIN_YEARS = 3


## ---------------------------------------------------------------------------
## Los métodos, todos con la misma firma: historia hasta el origen -> cuantiles
## ---------------------------------------------------------------------------

def analog_quantiles(history: pd.Series, map_name: str = "ols") -> np.ndarray:
    members, _ = build_ensemble(
        history.values[-SEARCH_YEARS * 8760:], window=WINDOW, horizon=HORIZON,
        k=K_ANALOGS, separation=SEPARATION, map_name=map_name,
    )
    return ensemble_quantiles(members, LEVELS)


def statsforecast_quantiles(history: pd.Series, model) -> np.ndarray:
    """Cuantiles de un modelo de statsforecast a partir de sus niveles."""
    from statsforecast import StatsForecast

    window = history.iloc[-BENCH_HISTORY_DAYS * 24:]
    frame = pd.DataFrame({"unique_id": "node", "ds": window.index,
                          "y": window.values})
    intervals = sorted({int(round(abs(100 * (2 * level - 1))))
                        for level in LEVELS if level != 0.5})
    forecast = StatsForecast(models=[model], freq="h", n_jobs=1).forecast(
        df=frame, h=HORIZON, level=intervals)

    name = repr(model)
    rows = []
    for level in LEVELS:
        if level == 0.5:
            rows.append(forecast[name].values)
        else:
            interval = int(round(abs(100 * (2 * level - 1))))
            side = "lo" if level < 0.5 else "hi"
            rows.append(forecast[f"{name}-{side}-{interval}"].values)
    return np.sort(np.vstack(rows), axis=0)


def seasonal_naive_quantiles(history: pd.Series) -> np.ndarray:
    from statsforecast.models import SeasonalNaive
    return statsforecast_quantiles(history, SeasonalNaive(season_length=24))


def theta_quantiles(history: pd.Series) -> np.ndarray:
    from statsforecast.models import DynamicOptimizedTheta
    return statsforecast_quantiles(history, DynamicOptimizedTheta(season_length=24))


def arima_quantiles(history: pd.Series) -> np.ndarray:
    from statsforecast.models import AutoARIMA
    return statsforecast_quantiles(history, AutoARIMA(season_length=24))


class QuantileBoosting:
    """Un modelo de bosque impulsado por cuantil, con horizonte directo."""

    LAGS = [1, 2, 3, 24, 25, 48, 72, 96, 120, 144, 168]

    def __init__(self, levels=LEVELS):
        self.levels = levels
        self.models = {}
        self.features = []

    @staticmethod
    def _design(series: pd.Series, origins: pd.DatetimeIndex) -> pd.DataFrame:
        rows = []
        for origin in origins:
            past = series.loc[:origin]
            if len(past) < 200:
                continue
            lags = {f"lag{lag}": past.iloc[-lag] for lag in QuantileBoosting.LAGS}
            for step in range(1, HORIZON + 1):
                target_time = origin + pd.Timedelta(hours=step)
                rows.append({**lags, "step": step, "hour": target_time.hour,
                             "dow": target_time.dayofweek,
                             "month": target_time.month,
                             "ds": target_time, "origin": origin})
        return pd.DataFrame(rows)

    def fit(self, series: pd.Series, origins: pd.DatetimeIndex) -> "QuantileBoosting":
        import lightgbm as lgb

        design = self._design(series, origins)
        design = design[design["ds"].isin(series.index)]
        target = series.loc[design["ds"]].values
        self.features = [c for c in design.columns if c not in ("ds", "origin")]
        for level in self.levels:
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=float(level), n_estimators=300,
                learning_rate=0.05, num_leaves=63, verbose=-1, n_jobs=4,
            )
            model.fit(design[self.features], target)
            self.models[float(level)] = model
        return self

    def predict(self, series: pd.Series, origin: pd.Timestamp) -> np.ndarray:
        design = self._design(series, pd.DatetimeIndex([origin]))
        rows = [self.models[float(level)].predict(design[self.features])
                for level in self.levels]
        return np.sort(np.vstack(rows), axis=0)


## ---------------------------------------------------------------------------
## Recorrido de orígenes
## ---------------------------------------------------------------------------

def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    series = load_series("pml_mda")

    origins = pd.date_range(TEST_START, TEST_END, freq=f"{ORIGIN_STEP_DAYS}D")
    origins = pd.DatetimeIndex([origin.replace(hour=23) for origin in origins])
    origins = origins[origins + pd.Timedelta(hours=HORIZON) <= series.index.max()]
    print(f"{len(origins)} orígenes  {origins[0]} -> {origins[-1]}", flush=True)

    training_origins = pd.date_range(
        origins[0] - pd.Timedelta(days=365 * LGBM_TRAIN_YEARS),
        origins[0] - pd.Timedelta(days=1), freq="2D")
    training_origins = pd.DatetimeIndex(
        [origin.replace(hour=23) for origin in training_origins])
    started = time.time()
    boosting = QuantileBoosting().fit(series, training_origins)
    print(f"bosque entrenado en {time.time() - started:.1f}s", flush=True)

    methods = {
        "Analog-Prob": lambda history: analog_quantiles(history, "ols"),
        "Analog-Raw": lambda history: analog_quantiles(history, "raw"),
        "SeasonalNaive": seasonal_naive_quantiles,
        "AutoARIMA": arima_quantiles,
        "Theta": theta_quantiles,
        "LGBM-Quantile": lambda history: boosting.predict(series, history.index[-1]),
    }

    fan_origin = origins[len(origins) // 2]
    rows, fan = [], []
    for number, origin in enumerate(origins, 1):
        history = series.loc[:origin]
        observed = series.loc[origin + pd.Timedelta(hours=1):
                              origin + pd.Timedelta(hours=HORIZON)].values
        if len(observed) != HORIZON:
            continue
        for name, method in methods.items():
            started = time.time()
            quantiles = method(history)
            elapsed = time.time() - started
            rows.append({"origin": origin, "method": name, "time_s": elapsed,
                         **evaluate(observed, quantiles)})
            if origin == fan_origin:
                for row, level in enumerate(LEVELS):
                    for step in range(HORIZON):
                        fan.append({"method": name, "level": level,
                                    "step": step + 1,
                                    "q": float(quantiles[row, step])})
        if number % 10 == 0:
            print(f"  origen {number}/{len(origins)}", flush=True)

    scores = pd.DataFrame(rows)
    scores.to_csv(RESULTS / "backtest_raw.csv", index=False)
    scores.groupby("method").mean(numeric_only=True).to_csv(
        RESULTS / "backtest_summary.csv")

    pd.DataFrame(fan).to_csv(RESULTS / "fan_example.csv", index=False)
    series.loc[fan_origin + pd.Timedelta(hours=1):
               fan_origin + pd.Timedelta(hours=HORIZON)].to_csv(
        RESULTS / "fan_actual.csv")
    with open(RESULTS / "config.json", "w") as handle:
        json.dump({"horizon": HORIZON, "window": WINDOW, "k": K_ANALOGS,
                   "separation": SEPARATION, "levels": LEVELS.tolist(),
                   "test_start": TEST_START, "test_end": TEST_END,
                   "n_origins": int(scores.origin.nunique()),
                   "fan_origin": str(fan_origin)}, handle, indent=2)

    print(scores.groupby("method").mean(numeric_only=True)[
        ["crps", "reliability", "sharp95", "cover95", "winkler95", "mae",
         "pinball95", "time_s"]].to_string())


if __name__ == "__main__":
    main()
