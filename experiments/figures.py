"""
Figuras del artículo, a partir de los CSV de results/.

Escribe PDF vectorial en paper/figs, que es de donde los toma el documento.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
FIGURES = BASE / "paper" / "figs"

METHODS = ["Analog-Prob", "Analog-Raw", "SeasonalNaive", "AutoARIMA",
           "Theta", "LGBM-Quantile"]
COLORS = dict(zip(METHODS, ["#1b4965", "#5fa8d3", "#9e9e9e", "#b5651d",
                            "#7a6c9b", "#2e7d5b"]))

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 120, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "serif",
})


def fan_chart() -> None:
    """Intervalos de un día, cuatro métodos, con lo observado encima."""
    fan = pd.read_csv(RESULTS / "fan_example.csv")
    actual = pd.read_csv(RESULTS / "fan_actual.csv")
    config = json.load(open(RESULTS / "config.json"))
    observed = actual.iloc[:, 1].values
    steps = np.arange(1, len(observed) + 1)

    shown = ["Analog-Prob", "AutoARIMA", "LGBM-Quantile", "SeasonalNaive"]
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True,
                                sharey=True)
    for axis, method in zip(axes.ravel(), shown):
        rows = fan[fan.method == method]
        if rows.empty:
            continue
        table = rows.pivot(index="step", columns="level", values="q")
        for low, high, alpha in [(0.05, 0.95, 0.15), (0.10, 0.90, 0.2),
                                 (0.25, 0.75, 0.3)]:
            axis.fill_between(table.index, table[low], table[high],
                              color=COLORS[method], alpha=alpha, lw=0)
        axis.plot(table.index, table[0.5], color=COLORS[method], lw=1.2,
                  label="median")
        axis.plot(steps, observed, color="black", lw=1.2, ls="--",
                  label="observed")
        axis.set_title(method, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=7)
    for axis in axes[1]:
        axis.set_xlabel("lead time (h)")
    for axis in axes[:, 0]:
        axis.set_ylabel("LMP (MXN/MWh)")
    figure.suptitle(
        f"Day-ahead predictive intervals, origin {config['fan_origin']}",
        fontsize=9)
    figure.savefig(FIGURES / "fan.pdf")
    plt.close(figure)


def reliability_diagram() -> None:
    """Frecuencia observada contra nivel nominal, un trazo por método."""
    scores = pd.read_csv(RESULTS / "backtest_raw.csv")
    columns = [c for c in scores.columns if c.startswith("hit_")]
    levels = np.array([float(c.split("_")[1]) for c in columns])

    figure, axis = plt.subplots(figsize=(3.4, 3.2))
    axis.plot([0, 1], [0, 1], color="black", lw=0.8, ls=":")
    for method in METHODS:
        rows = scores[scores.method == method]
        if rows.empty:
            continue
        axis.plot(levels, rows[columns].mean().values, marker="o", ms=3, lw=1,
                  color=COLORS[method], label=method)
    axis.set_xlabel("nominal level $\\alpha$")
    axis.set_ylabel("empirical frequency")
    axis.legend(frameon=False, fontsize=6.5, loc="upper left")
    figure.savefig(FIGURES / "reliability.pdf")
    plt.close(figure)


def score_and_time() -> None:
    """Las dos columnas que deciden si un método se puede operar."""
    scores = pd.read_csv(RESULTS / "backtest_raw.csv")
    average = scores.groupby("method")[["crps", "time_s"]].mean()
    average = average.reindex([m for m in METHODS if m in average.index])

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].barh(average.index, average["crps"],
                 color=[COLORS[m] for m in average.index])
    axes[0].set_xlabel("CRPS (MXN/MWh)")
    axes[0].invert_yaxis()
    axes[1].barh(average.index, average["time_s"],
                 color=[COLORS[m] for m in average.index])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("time per forecast (s, log scale)")
    axes[1].invert_yaxis()
    axes[1].set_yticklabels([])
    figure.savefig(FIGURES / "crps_time.pdf")
    plt.close(figure)


def score_over_time() -> None:
    """Pérdida por origen, para ver si la ventaja viene de unos pocos días."""
    scores = pd.read_csv(RESULTS / "backtest_raw.csv", parse_dates=["origin"])
    figure, axis = plt.subplots(figsize=(7.2, 2.4))
    for method in ["Analog-Prob", "AutoARIMA", "LGBM-Quantile"]:
        rows = scores[scores.method == method].sort_values("origin")
        axis.plot(rows.origin, rows.crps, lw=0.9, color=COLORS[method],
                  label=method)
    axis.set_yscale("log")
    axis.set_ylabel("CRPS (log)")
    axis.legend(frameon=False, fontsize=7, ncol=3)
    figure.savefig(FIGURES / "crps_time_series.pdf")
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fan_chart()
    reliability_diagram()
    score_and_time()
    score_over_time()
    print("figuras escritas en", FIGURES)


if __name__ == "__main__":
    main()
