# Testing the CVaR Model Against a Known Answer

A self-contained study evaluating the Value-at-Risk / expected-shortfall model for our
trend-following strategy, in two halves: synthetic processes where the true answer is known
exactly, and a trend-following backtest rebuilt from public futures prices.

This stands on its own. It is not a status update on the Phase 1 build and does not depend on
the earlier findings notebook.

**Start with `research_notebook.ipynb`.** Everything else here is the code and data behind it.

## Headline results

1. **A normal quantile has a fixed tail shape.** The ratio of a normal-based estimator's 99%
   number to its 90% number is 1.82 regardless of the data. On processes matching this
   strategy's measured parameters the true ratio is 2.18. The 17% shortfall produces a
   recognisable signature: comfortable at 90%, breaking through at 99%.
2. **On a clean long/short book, both the original model and its replacement over-breach at
   every level** — roughly double the expected rate at 99%. The cause is not tail shape, which
   is within 5%, but scale: the models are about 25% too narrow, persistently.
3. **A trailing scale recalibration fixes it out of sample.** One parameter, chosen on a
   training half, validated on a held-out half it never saw.

The shortfall is a portfolio-level effect: each market's own volatility is already filtered, so
what is missed is how much the markets move together — which is what a trend book is
concentrated in.

## Layout

```
research_notebook.ipynb     the study
README.md                   this file
code/                       everything runnable
data/                       inputs and saved outputs
figs/                       figures
```

### `code/` — the synthetic half

| file | what it does |
|---|---|
| `dgp.py` | The four process families (normal, Student-t, Hansen skewed-t, GARCH), each standardised to unit unconditional variance, plus the parameter grid. |
| `truth.py` | Ground truth: analytic VaR/CVaR where it exists, quadrature for the skewed-t tail mean, Monte Carlo for the GARCH marginal. Conditional and unconditional are kept distinct throughout. |
| `var_estimators.py` | Six estimators: naive historical, 4-sigma, moving average, EWMA, GARCH-normal, GARCH-t. |
| `grid.py` | The grid runner. Metrics computed per replication then averaged, so the standard errors are honest. |
| `analyse_grid.py` | Figures `mc_fig1-5` plus the crossover tables. Deliberately not a ranking — different methods fail in different places. |
| `transfer_test.py` | Matches the synthetic grid to the real data's measured parameters and compares predicted against actual coverage. |
| `validate_steps123.py` | 15 checks on the machinery. Run this first; nothing downstream means anything if it fails. |

### `code/` — the real-data half

| file | what it does |
|---|---|
| `backtest.py` | Rebuilds a trend-following backtest from public prices: Carver EWMAC, 34 markets, 1990–2024. Saves lag-correct position panels so nothing downstream has to shift an index with gaps in it. |
| `coverage_clean.py` | Walk-forward coverage test of both models on that backtest. Carries an identity check tying the exposure vector to the realised loss. |
| `diagnose.py` | Why both models fail, with the two tempting-but-invalid measurements named and avoided. |
| `portfolio_layer.py` | The scale correction, with every choice made on a training half and the held-out half reported. |
| `make_figs.py` | Figures `val_fig1-3`. |
| `paths.py` | Data-file resolution so the scripts run from anywhere. |
| `build_notebook.py` | Builds the notebook. **Edit this, not the `.ipynb`.** |

## Running it

```bash
cd code
python validate_steps123.py     # 15 checks — do this first
python backtest.py              # rebuild the backtest (fast)
python coverage_clean.py        # the coverage test (~5 min)
python diagnose.py              # the diagnosis
python portfolio_layer.py       # the correction
python analyse_grid.py          # synthetic figures
python make_figs.py             # real-data figures
```

`grid.py` regenerates `data/grid_results.csv` and takes considerably longer; the saved output is
included so the rest runs without it.

Needs `numpy`, `pandas`, `scipy`, `matplotlib`. No other dependencies.

## Data provenance

Prices come from [pysystemtrade](https://github.com/robcarver17/pysystemtrade), Rob Carver's
public repository — the same author as the book this strategy's rules follow. Daily closes
resampled from intraday bars; 34 of the strategy's 36 markets are present (Brazilian real and
Kansas wheat are not).

**Why not the original backtest database:** it is a collection of unverified runs of mixed
provenance. A quality screen — flat account-value histories, inconsistent position-to-equity
dating, blended strategies with nothing in the schema to flag them — left 18 of 549 runs usable
and exactly one holding short positions, with 27 position records. A trend-following strategy is
long/short by definition, so validating its risk model on an effectively long-only sample tests
something other than the strategy.

## Limitations

- The trend rule is a **reimplementation** of the published strategy (0.964 correlation with the
  observed production signals), not the production code. Conclusions about estimator behaviour
  transfer; exact risk numbers for the live book do not.
- **No transaction costs**, so the backtest's Sharpe is optimistic. This does not affect the tail
  behaviour under test.
- **One split**, 2010 onward, not rolling-origin validation.
- **Only the one-day horizon is tested.** 167 independent monthly observations give a 99% test no
  power, so nothing here supports or refutes the model at longer horizons.
- **Positions are held fixed within the day.** Correct for a strategy that does not re-trade
  intraday, but it means nothing here speaks to whipsaw.
