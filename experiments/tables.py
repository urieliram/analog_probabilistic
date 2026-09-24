"""
Cuerpos de cuadro y macros del artículo, a partir de los CSV de results/.

Ninguna cifra del artículo se escribe a mano: todas salen de aquí, a
paper/tables. Las macros de LaTeX no admiten dígitos en el nombre, de ahí los
nombres deletreados.
"""

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
TABLES = BASE / "paper" / "tables"

METHOD_ORDER = ["Analog-Prob", "Analog-Raw", "SeasonalNaive", "Theta",
                "AutoARIMA", "LGBM-Quantile"]
METHOD_LABEL = {
    "Analog-Prob": "Analog-Prob (this work)",
    "Analog-Raw": "Analog ensemble, $b_i=1$ (ablation)",
    "SeasonalNaive": "Seasonal na\\\"ive",
    "Theta": "Dynamic optimized Theta",
    "AutoARIMA": "AutoARIMA$(p,d,q)(P,D,Q)_{24}$",
    "LGBM-Quantile": "LightGBM quantile regression",
}
VARIANT_ORDER = ["raw", "rescale", "ols", "ols-weighted", "ols-noclip"]
VARIANT_LABEL = {
    "raw": "$b_i=1$ (analog ensemble)",
    "rescale": "$b_i=\\sigma_Y/\\sigma_{X_i}$ (pattern rescaling)",
    "ols": "$b_i=\\rho_i\\sigma_Y/\\sigma_{X_i}$ (this work)",
    "ols-weighted": "\\quad with similarity weights",
    "ols-noclip": "\\quad without output clipping",
}


def number(value: float, decimals: int = 1) -> str:
    """Número con separador de miles fino, como lo escribe LaTeX."""
    return f"{value:,.{decimals}f}".replace(",", "\\,")


def write(name: str, lines: list) -> None:
    (TABLES / name).write_text("\n".join(lines) + "\n")


def main_table(average: pd.DataFrame) -> None:
    best = {column: average[column].min()
            for column in ["crps", "reliability", "winkler95", "mae", "time_s"]}
    lines = []
    for method, row in average.iterrows():
        cells = [METHOD_LABEL[method]]
        for column, decimals in [("crps", 1), ("reliability", 3), ("sharp80", 0),
                                 ("sharp95", 0), ("cover80", 3), ("cover95", 3),
                                 ("winkler95", 0), ("mae", 1), ("time_s", 3)]:
            cell = number(row[column], decimals)
            if column in best and abs(row[column] - best[column]) < 1e-12:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    write("main_results.tex", lines)


def tail_table(average: pd.DataFrame) -> None:
    best = average["pinball95"].min()
    lines = []
    for method, row in average.iterrows():
        cell = number(row["pinball95"], 2)
        if abs(row["pinball95"] - best) < 1e-12:
            cell = f"\\textbf{{{cell}}}"
        lines.append(
            f"{METHOD_LABEL[method]} & {cell} & {number(100 * row['hit_0.95'], 1)}"
            f" & {number(100 * (1 - row['cover95']), 1)}"
            f" & {number(row['rmse'], 1)} \\\\")
    write("tail_results.tex", lines)


def dm_table() -> None:
    tests = pd.read_csv(RESULTS / "dm_test.csv")
    lines = []
    for _, row in tests.iterrows():
        p_value = ("$<$0.0001" if row["p_value"] < 1e-4
                   else f"{row['p_value']:.4f}")
        lines.append(f"{METHOD_LABEL[row['method']]} & "
                     f"{number(row['mean_diff'], 2)} & "
                     f"{number(row['dm_stat'], 2)} & {p_value} \\\\")
    write("dm_results.tex", lines)


def ablation_table(ablation: pd.DataFrame) -> None:
    lines = []
    for variant in VARIANT_ORDER:
        if variant not in ablation.index:
            continue
        row = ablation.loc[variant]
        lines.append(
            f"{VARIANT_LABEL[variant]} & {number(row['crps'], 1)} & "
            f"{number(row['reliability'], 3)} & {number(row['sharp95'], 0)} & "
            f"{number(row['cover95'], 3)} & {number(row['pinball95'], 2)} & "
            f"{number(row['mae'], 1)} \\\\")
    write("ablation_results.tex", lines)


def calibration_table(calibration: pd.DataFrame) -> None:
    names = {"raw": "Ensemble quantiles (as proposed)",
             "calibrated": "Recalibrated on past origins"}
    lines = []
    for variant in ["raw", "calibrated"]:
        row = calibration.loc[variant]
        lines.append(
            f"{names[variant]} & {number(row['crps'], 1)} & "
            f"{number(row['reliability'], 3)} & {number(row['sharp95'], 0)} & "
            f"{number(row['cover80'], 3)} & {number(row['cover95'], 3)} & "
            f"{number(row['winkler95'], 0)} & {number(row['pinball95'], 2)} \\\\")
    write("calibration_results.tex", lines)


def sensitivity_table(grid: pd.DataFrame) -> None:
    best = grid["crps"].min()
    lines = []
    for _, row in grid.iterrows():
        crps = number(row["crps"], 1)
        if abs(row["crps"] - best) < 1e-12:
            crps = f"\\textbf{{{crps}}}"
        lines.append(f"{int(row['window'])} & {int(row['k'])} & "
                     f"{row['separation']:.1f} & {crps} & "
                     f"{number(row['reliability'], 3)} & "
                     f"{number(row['cover95'], 3)} & "
                     f"{number(row['time_s'], 2)} \\\\")
    write("sensitivity_results.tex", lines)


def macros(average: pd.DataFrame, scores: pd.DataFrame, config: dict,
           ablation: pd.DataFrame, calibration: pd.DataFrame,
           grid: pd.DataFrame, clip_rate: float) -> None:
    analog = average.loc["Analog-Prob"]
    arima = average.loc["AutoARIMA"]
    boosting = average.loc["LGBM-Quantile"]
    naive = average.loc["SeasonalNaive"]
    best_grid = grid.loc[grid["crps"].idxmin()]
    chosen = grid[(grid.window == config["window"]) & (grid.k == config["k"])
                  & (grid.separation == config["separation"])]

    values = {
        "NORIGINS": str(int(scores.origin.nunique())),
        "NFORECASTS": str(int(scores.origin.nunique() * config["horizon"])),
        "TESTSTART": config["test_start"],
        "TESTEND": config["test_end"],
        "KANALOGS": str(config["k"]),
        "VSELE": str(config["window"]),
        "HORIZON": str(config["horizon"]),
        "APCRPS": number(analog["crps"], 1),
        "APREL": number(analog["reliability"], 3),
        "APTIME": number(analog["time_s"], 3),
        "APCOVNINETYFIVE": number(100 * analog["cover95"], 1),
        "APCOVEIGHTY": number(100 * analog["cover80"], 1),
        "APWINK": number(analog["winkler95"], 0),
        "APMAE": number(analog["mae"], 1),
        "APPINNINETYFIVE": number(analog["pinball95"], 2),
        "APSHARPNINETYFIVE": number(analog["sharp95"], 0),
        "SNREL": number(naive["reliability"], 3),
        "SNSHARPNINETYFIVE": number(naive["sharp95"], 0),
        "SNCOVNINETYFIVE": number(100 * naive["cover95"], 1),
        "LGBMCOVNINETYFIVE": number(100 * boosting["cover95"], 1),
        "LGBMTIME": number(boosting["time_s"], 3),
        "ARIMATIME": number(arima["time_s"], 1),
        "SPEEDUP": number(arima["time_s"] / analog["time_s"], 0),
        "CRPSGAINARIMA": number(
            100 * (arima["crps"] - analog["crps"]) / arima["crps"], 1),
        "CRPSGAINLGBM": number(
            100 * (boosting["crps"] - analog["crps"]) / boosting["crps"], 1),
        "CRPSGAINRAW": number(
            100 * (average.loc["Analog-Raw", "crps"] - analog["crps"])
            / average.loc["Analog-Raw", "crps"], 1),
        "CLIPRATE": number(100 * clip_rate, 2),
        "NCAL": str(int(calibration.attrs["origins"])),
        "CALCOV": number(100 * calibration.loc["calibrated", "cover95"], 1),
        "CALCOVRAW": number(100 * calibration.loc["raw", "cover95"], 1),
        "CALCRPS": number(calibration.loc["calibrated", "crps"], 1),
        "CALCRPSRAW": number(calibration.loc["raw", "crps"], 1),
        "CALSHARP": number(calibration.loc["calibrated", "sharp95"], 0),
        "CALSHARPRAW": number(calibration.loc["raw", "sharp95"], 0),
        "CALWINK": number(calibration.loc["calibrated", "winkler95"], 0),
        "CALWINKRAW": number(calibration.loc["raw", "winkler95"], 0),
        "NSENS": str(int(grid["n_origins"].max())),
        "SENSBESTV": str(int(best_grid["window"])),
        "SENSBESTK": str(int(best_grid["k"])),
        "SENSBESTTOL": f"{best_grid['separation']:.1f}",
        "SENSBESTCRPS": number(best_grid["crps"], 1),
        "SENSTMIN": number(grid["time_s"].min(), 3),
        "SENSTMAX": number(grid["time_s"].max(), 3),
    }
    for variant, tag in [("raw", "ANEN"), ("rescale", "DUD"), ("ols", "OLS"),
                         ("ols-noclip", "NOCLIP")]:
        if variant in ablation.index:
            row = ablation.loc[variant]
            values[f"AB{tag}CRPS"] = number(row["crps"], 1)
            values[f"AB{tag}COV"] = number(100 * row["cover95"], 1)
            values[f"AB{tag}REL"] = number(row["reliability"], 3)
            values[f"AB{tag}SHARP"] = number(row["sharp95"], 0)
            values[f"AB{tag}PIN"] = number(row["pinball95"], 2)
    values["ABOLSGAINANEN"] = number(
        100 * (ablation.loc["raw", "crps"] - ablation.loc["ols", "crps"])
        / ablation.loc["raw", "crps"], 1)
    if len(chosen):
        values["SENSAPRIORI"] = number(chosen.iloc[0]["crps"], 1)
        values["SENSGAP"] = number(
            100 * (chosen.iloc[0]["crps"] - best_grid["crps"])
            / chosen.iloc[0]["crps"], 1)

    write("macros.tex", [f"\\newcommand{{\\{name}}}{{{value}}}"
                         for name, value in values.items()])


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    config = json.load(open(RESULTS / "config.json"))

    scores = pd.read_csv(RESULTS / "backtest_raw.csv", parse_dates=["origin"])
    average = scores.groupby("method").mean(numeric_only=True)
    average = average.reindex([m for m in METHOD_ORDER if m in average.index])

    ablation_rows = pd.read_csv(RESULTS / "ablation_raw.csv")
    ablation = ablation_rows.groupby("variant").mean(numeric_only=True)

    calibration_rows = pd.read_csv(RESULTS / "calibration_raw.csv")
    calibration = calibration_rows.groupby("variant").mean(numeric_only=True)
    calibration.attrs["origins"] = calibration_rows.origin.nunique()

    grid = pd.read_csv(RESULTS / "sensitivity.csv")
    clip_rate = float(pd.read_csv(RESULTS / "clip_rate.csv",
                                  index_col=0).iloc[0, 0])

    main_table(average)
    tail_table(average)
    dm_table()
    ablation_table(ablation)
    calibration_table(calibration)
    sensitivity_table(grid)
    macros(average, scores, config, ablation, calibration, grid, clip_rate)

    print(average[["crps", "reliability", "sharp95", "cover95", "winkler95",
                   "mae", "pinball95", "time_s"]].to_string())


if __name__ == "__main__":
    main()
