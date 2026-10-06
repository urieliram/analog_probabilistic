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

## In plain words

The method looks through the history for the **episodes** when the price behaved
like it does now, sees what happened next in each one, and uses those continuations
as tomorrow's possible futures. Forty futures, not a single **point forecast**.
Together they make a fan: how cheap it could come out at the bottom, how expensive
at the top.

Each past continuation has to be scaled to the present before it is used, and that
is where a decision hides. You can leave its spread alone, or you can pull the
continuations in toward their average. Pulling them in is what least squares does,
and it is what the earlier version of this work presented as its contribution. How
hard you pull is one number, and it has a name: the **shrinkage exponent** γ. At γ = 0
the spread is left alone; at γ = 1 you get the least-squares slope.

**Pulling them in is wrong.** It closes the fan, and a closed fan promises less risk
than there is: the real price lands above the top more often. That held over six
years, 25 load zones, always in the same direction, without a single exception —
between seven and ten extra times in every hundred.

What pulling in did buy was a slightly better centre: the point forecast landed
closer. So it looked like a trade — give up the fan, gain accuracy. **It is not.**
That accuracy shows up only in the calm year the method was tuned on. Over the five
following years it reverses, and it reverses harder the more volatile the year. In
2024, the worst of them, pulling in was worse on both counts at once.

There is no trade. Pulling in is simply worse.

This matters most to whoever uses the forecast to hedge rather than to rank models.
The expensive mistake is falling short of the peak, and a closed fan is exactly the
machine for falling short of the peak.

And the methodological point is the uncomfortable one: **tune on one year and call it
settled, and you will choose wrong.** We did, and the five years we had not touched
corrected us.

Same text in Spanish: [README_es.md](README_es.md).

## The member map

Writing each member as `y = a + b·x`, three slopes are of interest, and they
are three different methods from the literature:

| `map_name` | slope | what it is |
|---|---|---|
| `raw` | `b = 1` | analog ensemble, members untouched |
| `rescale` | `b = σ_Y / σ_X` | pattern rescaling |
| `ols` | `b = ρ · σ_Y / σ_X` | least squares, the rescaling shrunk by similarity |

They share the search, the horizon and the cost, so they can be compared
without confounding selection with the map. They are also three positions of one
family indexed by a **shrinkage exponent**,

```
b_i(γ) = ρ_i^γ · σ_Y / σ_{X_i}
```

with γ = 1 the least-squares slope and γ = 0 the pattern rescaling. The exponent
matters because the analogs were selected for having a high ρ, and using that same ρ
again to shrink the slope narrows an ensemble that was already too narrow: the
variance each member transmits is multiplied by ρ^(2γ). γ is chosen on the selection
stretch, not by the authors.

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

`data/panel/` holds the hourly day-ahead price of the **25 load zones** used in the
study, from April 2018 to today, as one gzipped CSV per zone with the price and its
three components: energy, losses and congestion. Prices are in Mexican pesos per
megawatt-hour and come from CENACE, the national energy control centre, which
publishes them for the 101 load zones of the national interconnected system.

The 101 zones are not 101 independent pieces of evidence. They belong to one
transmission network, and a single common factor — fuel cost and national demand —
explains 85% of their variation over the period used to pick the panel. The panel
is therefore chosen by grouping the zones on what is left after removing that
common factor, taking the centre of each group, and adding the extreme zones fixed
in advance by rule. `results/zone_groups.csv` lists all 101 zones with their group
and their representative, so nothing is hidden.

`experiments/build_panel.py` rebuilds the full 101-zone panel from the CENACE daily
archives; the complete panel is 107 MB and lives in an archived repository with a
permanent identifier rather than in git.

Three things are worth knowing before using the files. The index is in **local civil
time**, as the market speaks. Mexico kept daylight saving until late 2022, so the
five last Sundays of October from 2018 to 2022 have 25 hours in the published
report: the repeated hour is dropped, and the corresponding Sundays in April have
23. And `data/publication_stamps.csv` records, for each daily file, the hour at
which CENACE generated it.

## Protocol

The series is split into stretches that never mix backwards, fixed in
`experiments/protocol.py`:

| stretch | dates | what it is |
|---|---|---|
| search bank | the two years before each origin, rolling | where analogs come from |
| selection | 2020-04-01 to 2021-03-31 | where the configuration is chosen |
| primary test | 2021-04-01 to 2026-02-28 | never inspected |
| already inspected | 2026-03-01 onwards | reported separately |

The configuration is not written by hand anywhere: it is chosen on the selection
stretch and frozen in `results/selected_config.json`, which the experiment reads and
without which it refuses to run.

The forecast is issued on the morning of day *d*, before the 10:00 bid deadline, and
targets the 24 hours of day *d+1*, so lead times run from 15 to 38 hours. That is
not an assumption: `experiments/publication_times.py` measures, from the timestamp
each archive carries, that over eight and a half years the prices for the next day
were never generated before 13:57 — always after the bid window had closed.

## Reproducing the experiments

```bash
python experiments/build_panel.py      # rebuild the 101-zone panel from archives
python experiments/select_zones.py     # choose the panel, freeze it
python experiments/gamma_sweep.py      # shrinkage exponent on the selection stretch
python experiments/select_config.py    # apply the rule, freeze the configuration
python experiments/run_forecasts.py    # produce forecasts into the store
python experiments/evaluate.py results/forecasts_prueba.parquet
python experiments/tables.py           # paper/tables/*.tex
python experiments/figures.py          # paper/figs/*.pdf
```

Producing forecasts and scoring them are two separate steps. The run writes what it
produced — the quantiles, and the ensemble members where a method has them — and the
evaluation reads that file. Changing a metric then costs seconds instead of hours,
and every table in the paper comes from the same file.

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
