"""
Prueba de Diebold-Mariano del método propuesto contra cada comparativo.

Se aplica sobre la pérdida por origen, de una cola: la hipótesis alternativa es
que el análogo probabilístico pierde menos. Escribe results/dm_test.csv.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backtest import RESULTS  # noqa: E402
from experiments.scores import diebold_mariano  # noqa: E402

REFERENCE = "Analog-Prob"


def main() -> None:
    scores = pd.read_csv(RESULTS / "backtest_raw.csv", parse_dates=["origin"])
    losses = scores.pivot(index="origin", columns="method", values="crps").dropna()

    rows = []
    for method in losses.columns:
        if method == REFERENCE:
            continue
        difference = (losses[REFERENCE] - losses[method]).values
        statistic, p_value = diebold_mariano(difference)
        rows.append({"method": method, "dm_stat": statistic,
                     "p_value": p_value, "mean_diff": float(difference.mean())})

    table = pd.DataFrame(rows).sort_values("dm_stat")
    table.to_csv(RESULTS / "dm_test.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
