# analog_probabilistic

Probabilistic forecasting by analogs: code, data and experiments behind the
paper *Probabilistic Analog Forecasting of Day-Ahead Locational Marginal
Prices: An Ensemble of Pairwise Analog Regressions*.

The analog method searches the history for the stretches that most resemble the
present, fits a map between each of them and the present, and applies that map
to what followed each analog. Every analog contributes one complete future
trajectory; the ensemble of trajectories is the forecast, and its empirical
quantiles are the predictive distribution. No training stage, no distributional
assumption, no residual model.

## The member map

Writing each member as `y = a + b·x`, three slopes are of interest, and they
are three different methods from the literature:

| `map_name` | slope | what it is |
|---|---|---|
| `raw` | `b = 1` | analog ensemble, members untouched |
| `rescale` | `b = σ_Y / σ_X` | pattern rescaling |
| `ols` | `b = ρ · σ_Y / σ_X` | least squares, the rescaling shrunk by similarity |

They share the search, the horizon and the cost, so they can be compared
without confounding selection with the map.

## Install

```bash
pip install -r requirements.txt
```

## Use

```python
import pandas as pd
from analog_probabilistic import AnalogProbabilistic

prices = pd.read_csv("data/nodes/san_juan_del_rio.csv.gz",
                     parse_dates=["ds"]).set_index("ds")["pml_mda"]

model = AnalogProbabilistic(window=48, k=40).fit(prices.values)
quantiles = model.quantiles(horizon=24, levels=[0.05, 0.5, 0.95])
risk = model.exceedance(horizon=24, threshold=2000.0)
```

`quantiles` has one row per level and one column per lead time; `risk` is the
share of members above the threshold at each lead time, which is the form a
buyer uses.

## Data

`data/nodes/` holds the hourly price history of fourteen nodes of the Mexican
wholesale electricity market, one gzipped CSV per node, about 1.1 million
node-hours in total. `data/nodes_index.csv` lists them with their period and
size. Each file has the same eight columns: the locational marginal price and
its three components — energy, losses and congestion — for the day-ahead
(`_mda`) and the real-time (`_mtr`) markets, in Mexican pesos per megawatt-hour.

| node | kind | hours | period |
|---|---|---|---|
| `Mexican_Oriente` | market node | 93,422 | 2016-01-27 to 2026-09-23 |
| `05LGA-115`, `06CDU-400`, `06MES-400`, `06PAE-400` | market nodes | 90,858 each | 2016-01-29 to 2026-06-10 |
| `iguala`, `monclova`, `san_juan_del_rio` | zonal averages | 74,276 each | 2018-04-04 to 2026-09-23 |
| `juarez`, `leon`, `monterrey`, `piedras_negras`, `reynosa` | zonal averages | 71,756 each | 2018-04-04 to 2026-06-10 |
| `03POM-400` | market node | 49,510 | 2020-10-17 to 2026-06-10 |

The paper uses `san_juan_del_rio`, day-ahead price.

Three things are worth knowing before using the files. The day-ahead series
have a handful of missing hours caused by publication failures, which the
experiments fill by linear interpolation. The real-time series end earlier than
the day-ahead ones, because the operator publishes them with a lag. And the
collection for several nodes stopped in June 2026, so their series end there.

The source is CENACE, the national energy control centre, through its public
information system. Demand series are deliberately absent from this
repository.

## Reproducing the experiments

```bash
python experiments/backtest.py       # main comparison, ~40 min
python experiments/ablation.py       # the three member maps
python experiments/calibration.py    # recalibration on past origins
python experiments/sensitivity.py    # parameter grid
python experiments/dm_test.py        # significance
python experiments/tables.py         # paper/tables/*.tex
python experiments/figures.py        # paper/figs/*.pdf
```

`backtest.py` is what takes the time, and all but a minute of it goes to
AutoARIMA.
The results already in `results/` were produced on an Intel Core i7-11700 with
64 GB of RAM under Python 3.10.

## Results

Rolling origins every two days from 1 March to 31 August 2026, horizon 24 h,
92 origins, day-ahead prices. Scores in MXN/MWh, time in seconds per forecast.

| Method | CRPS | Δ reliability | Sharp. 95% | Cov. 95% | Winkler | MAE | Pinball 0.95 | Time |
|---|---|---|---|---|---|---|---|---|
| Analog-Prob | **62.3** | 0.216 | **515** | 0.778 | **1139** | **156.2** | **39.3** | 0.027 |
| Analog ensemble (`raw`) | 79.0 | 0.257 | 806 | 0.815 | 1591 | 197.2 | 62.8 | 0.020 |
| AutoARIMA | 80.2 | 0.210 | 816 | 0.907 | 1657 | 193.9 | 53.6 | 65.703 |
| Seasonal naive | 81.8 | **0.181** | 848 | 0.904 | 1945 | 173.4 | 51.4 | **0.017** |
| Theta | 98.3 | 0.215 | 1394 | 0.936 | 1884 | 208.0 | 43.1 | 0.171 |
| LightGBM quantile | 68.0 | 0.227 | 445 | 0.686 | 1349 | 171.2 | 47.5 | 0.044 |

Every difference in CRPS favours the analog method and is significant by a
one-sided Diebold-Mariano test; the closest, against LightGBM, gives p = 0.026.

Three findings deserve to be read together with the table:

1. **The intervals are too narrow.** The nominal 95% interval covers 77.8%.
   Members are chosen for resembling the present, so they resemble one another
   and the ensemble understates how much the future can differ. Recalibrating
   with the errors of past origins raises coverage to 89.4% and widens the
   intervals from 628 to 1059, leaving the CRPS untouched.
2. **The un-shrunk rescaling does better than least squares here**, 60.9
   against 62.1 in CRPS and 0.865 against 0.810 in coverage. Shrinking by
   similarity tightens an ensemble that is already too tight.
3. **The window matters more than the ensemble size.** In the parameter grid a
   one-week window with 20 analogs gives the best score; the configuration
   fixed in advance (48 hours, 40 analogs) lands 4.6% above it.

The analog method is also 2,400 times faster than AutoARIMA, which it beats on
every score. The search compares the present window against every candidate in
one matrix product; the loop-based version used while the method was developed
returns the same ensemble to within 5e-13 and takes about fifty times longer.

## Paper

`paper/` holds the LaTeX source, the bibliography, the generated tables and the
figures. Every number in the paper is written by `experiments/tables.py`; none
is typed by hand.

## Citing

The method in its point form is described in:

> U. I. Lezama-Lope, A. Benavides-Vázquez, G. Santamaría-Bonfil and
> R. Z. Ríos-Mercado, "Fast and Efficient Very Short-Term Load Forecasting
> Using Analogue and Moving Average Tools", *IEEE Latin America
> Transactions*, vol. 21, no. 9, pp. 1015–1021, 2023.

The probabilistic version is the subject of the paper in `paper/`, currently
under preparation.

## License

Apache License 2.0. The price data is public information published by CENACE.
