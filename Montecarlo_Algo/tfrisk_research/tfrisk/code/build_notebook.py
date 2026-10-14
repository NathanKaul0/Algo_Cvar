"""Builds research_notebook.ipynb. Edit this file, not the .ipynb.

Audience: the overseer, in a meeting. Structure follows that - findings and
the decisions they force come first, method second, caveats stated rather than
buried. Scope is the Monte Carlo work only: the synthetic study and the
public-data validation built on top of it.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": t.strip("\n").split("\n")})


def code(src, stdout=None, figure=None):
    outs = []
    if stdout:
        outs.append({"output_type": "stream", "name": "stdout",
                     "text": [l + "\n" for l in stdout.strip("\n").split("\n")]})
    if figure:
        outs.append({"output_type": "display_data",
                     "data": {"text/plain": [f"<figure: {figure}>"]},
                     "metadata": {}})
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": outs,
                  "source": src.strip("\n").split("\n")})


def fig(name, caption):
    md(f"![{caption}](figs/{name})\n\n*{caption}*")


# ===========================================================================
md("""
# Testing the CVaR Model Against a Known Answer

**Monte Carlo study on synthetic data, then validation on a trend-following backtest rebuilt
from public prices.**

Nate Kaul · AlgoGators Quant Trading · October 2026

---

### What this covers

Two days of work, standing on its own. It is not a status update on the Phase 1 build and does
not depend on the earlier findings notebook.

The problem it set out to solve: **on real data you never know the right answer.** If a model's
99% estimate gets breached on 2% of days, that could mean the model understates risk by half, or
it could mean the sample was unlucky — and nothing in the data separates the two. So the model
was tested somewhere the answer *is* known, and then the predictions that test produced were
checked against real data.

Along the way the second problem became unavoidable: the backtest database cannot support a
coverage test, so a clean one was rebuilt from public futures prices. That rebuild is the first
time this model has been scored on a genuinely long/short book.
""")

md("""
---

## The three findings

**1. Any estimator that reads its quantile off a normal distribution has a fixed tail shape.**

The ratio of its 99% number to its 90% number is locked at 1.82 — a property of the formula,
not of the data. On processes carrying this strategy's measured parameters the true ratio is
**2.18**. That 17% shortfall produces a specific signature: *comfortable at the 90% level,
breaking through at the 99%.* Only two of six estimators get the shape right.

**2. On a clean long/short book, both our models over-breach at every level.**

Roughly double the expected rate at 99%, for both the original model and its replacement, over
7,305 test days. The cause is not tail shape — that is within 5% — but **scale: the models are
about 25% too narrow, persistently.**

**3. A trailing scale correction fixes it, and it holds on data it never saw.**

One parameter, selected on a training half, validated on a held-out half. The pre-registered
choice gets the body right and the tail marginally wrong; slightly longer windows pass at all
three levels.

The shortfall is a **portfolio-level** effect. Each market's own volatility is already filtered
by its own GARCH, so what is missing is how much the markets move *together* — which is exactly
what a trend book is concentrated in, because the rules put on correlated bets by construction.
""")

md("""
---

## What this forces us to decide

Five things, in the order they matter. Each is a decision for the team, not a task I can close
alone.

| # | Decision | What the evidence says |
|---|---|---|
| 1 | **Adopt the scale correction?** | The model is 25% too narrow on clean data. The correction is one trailing parameter and survives out-of-sample validation. Recommend adopting, with a 1,000-day window. |
| 2 | **Keep or retire the correlation regime?** | The plan of record requires a regime-switching correlation state. Measured dependence in this book does not exceed an elliptical null, so a *regime* is not the right shape. But the portfolio-level effect the regime was meant to capture is real and is what finding 2 identifies. Recommend: satisfy the requirement with the scale layer, not a regime, and record it as a deviation. |
| 3 | **Drop the annual CVaR figure?** | It cannot be validated and will not be. Recommend dropping it rather than reporting an unvalidated number. |
| 4 | **Accept public-price validation as standing practice?** | The backtest database leaves 18 of 549 runs usable and one holding shorts. Public prices give 8,565 clean days with 37.5% short exposure. Recommend adopting as the standing validation substrate until the database is fixed. |
| 5 | **Who owns the four data questions?** | Why most configurations never go short, why most equity histories are flat, which runs are single-strategy, and why position-to-equity dating is inconsistent between runs. These block validation on our own data permanently if unanswered. |
""")

md("""
---

## Part I — Synthetic data, where the answer is known

### The design

Four process families, each standardised to unit unconditional variance, so differences between
them are differences of *shape* and never of scale:

| family | what it isolates |
|---|---|
| normal | the control — no fat tails, no skew, no clustering |
| Student-t | tail thickness alone, indexed by tail index ν |
| Hansen skewed-t | asymmetry, with the left tail fattened independently |
| GARCH(1,1) | volatility clustering, persistence α+β on a dial |

Six estimators are scored against every cell — naive historical, a 4-sigma rule, a moving
average, EWMA, GARCH-normal, GARCH-t — across four estimation windows and three confidence
levels.

**Indexed by tail index, not kurtosis.** Our measured per-market tail indices run from 2.7 to
18.9, and kurtosis does not exist below ν = 4. Indexing by kurtosis would have silently dropped
the regime where naive methods fail hardest.

Ground truth is the entire point of the exercise, so none of it is taken on faith.
""")

code('''
# 15 checks: the processes are what they claim to be, the truth is analytically
# correct AND matches a 4-million-draw Monte Carlo, the estimators are exact on
# the control case. Nothing downstream means anything if these fail.
!python validate_steps123.py
''', stdout="""
A. PROCESSES - standardisation and moments
  [PASS] every process is variance-stationary (unconditional var = 1)
  [PASS] sample mean ~0 (stable 4th moment, persistence < 0.97)   worst 0.0047
  [PASS] sample variance ~1 (stable 4th moment, persistence < 0.97)  worst 0.0188
  [PASS] skew-t with lam=0 == standardised t          max diff = 0.0e+00
  [PASS] skew-t density integrates to 1, mean 0, var 1   worst dev 3.5e-05
  [PASS] skew sign follows lam
  [PASS] GARCH theoretical kurtosis matches simulation
  [PASS] every cell has a defined tail index

B. GROUND TRUTH - analytic vs 4M-draw Monte Carlo
  [PASS] analytic VaR matches Monte Carlo             worst gap = 0.0080
  [PASS] analytic CVaR matches Monte Carlo            worst gap = 0.0096
  [PASS] GARCH conditional VaR breaches at exactly 1-q   worst rel error 1.3%
  [PASS] conditional VaR genuinely varies under GARCH    99th/1st = 2.30x

C. ESTIMATORS - the control case (normal data, known answer)
  [PASS] all estimators within 3% of truth on normal data   worst bias 1.02%
  [PASS] EWMA sd matches its effective sample size
  [PASS] 4-sigma implied confidence computed

  15/15 checks passed
""")

md("""
### Finding 1 in detail: the tail-shape ceiling

The ratio of an estimator's 99% number to its 90% number is a scale-free measure of tail
curvature. For anything reading its quantile off a normal distribution:

$$\\frac{\\Phi^{-1}(0.99)}{\\Phi^{-1}(0.90)} = \\frac{2.326}{1.282} = 1.82$$

No amount of data moves it. On a process carrying our measured parameters — tail index 6.9,
skewness −0.21, persistence 0.984 — the truth is 2.18.

| estimator | shape ratio | verdict |
|---|---|---|
| moving average, EWMA, GARCH-normal | 1.82 at every window | flat by 17% |
| GARCH-t | 2.08 – 2.11 | close |
| naive historical | 2.29 – 2.48 | slightly over-steep |

This is why it matters operationally: an estimator whose tail is too flat is *generous in the
body and short in the extreme*. It under-breaches at 90% and over-breaches at 99%. That pattern
is recognisable in real coverage output, and it tells you the problem is shape rather than level.
""")

fig("mc_fig1_kurtosis_bias.png",
    "Fat tails: a normal quantile understates VaR, and the gap widens as the tail thickens")
fig("mc_fig2_kurtosis_breach.png",
    "The operational consequence — how often losses break through the estimate")

md("""
### Where each method breaks

The output is deliberately **not a ranking**. Different methods fail in different places, and a
ranking would hide exactly that.

| question | answer from the grid |
|---|---|
| When does a reactive volatility model beat a flat window? | Only once persistence exceeds ~0.9. Below that the flat window wins on RMSE. |
| When does assuming a normal tail stop being acceptable? | Below tail index ≈ 8 the 99% breach rate passes 1.25%; below 5 it fails outright. |
| How much history before the empirical method overtakes the parametric one? | Around 1,000 days. Below that GARCH-t wins; above it naive historical does. |
| What does over-fitting cost? | GARCH-t on data with no fat tails and no clustering carries a 30–70% RMSE penalty versus a moving average. |
| What does skewness do? | Every symmetric method fails. At λ = −0.6, GARCH-t is 21% low and a moving average 27% low. Naive historical is within 4%, because it never assumed symmetry. |
""")

fig("mc_fig3_clustering_rmse.png",
    "Clustering: where a reactive volatility model earns its keep")
fig("mc_fig4_skew_bias.png",
    "Skewness: every symmetric method fails; the empirical one does not")

md("""
---

## Part II — A backtest worth testing against

### Why we could not use our own

The coverage test needs a backtest to score against. Ours is a collection of unverified runs of
mixed provenance. A quality screen — flat account-value histories, inconsistent
position-to-equity dating, blended strategies with nothing in the schema to flag them — left
**18 of 549 runs usable, and exactly one holding short positions, with 27 position records.**

A trend-following strategy is long/short by definition. Validating its risk model on an
effectively long-only sample tests something other than the strategy.

### What we used instead

Prices from [pysystemtrade](https://github.com/robcarver17/pysystemtrade), Rob Carver's public
repository — the same author as the book our rules follow. **34 of our 36 markets are present**
(Brazilian real and Kansas wheat are not). Daily closes resampled from intraday bars give
**8,565 trading days, 1990 to 2024.**

The rule is our own reimplemented EWMAC: three speed pairs, Carver's forecast scalars, ±20 caps,
FDM 1.45, 20% volatility target, 10% position buffer, and an instrument diversification
multiplier re-estimated quarterly on trailing data.

**One technical point worth making to anyone who asks about contract specs.** Back-adjusted
futures prices can cross zero, so this works in price differences normalised by a volatility in
the same units. With

$$\\text{contracts}_i = \\frac{f_i}{10}\\cdot\\frac{C\\cdot\\tau}{N\\,\\sigma_i}, \\qquad
\\text{P\\&L}_i = \\text{contracts}_i \\cdot \\Delta p_i$$

the price units cancel exactly — **contract multipliers and quote currencies drop out of the
risk-targeted P&L.** Their absence cannot bias this backtest.
""")

code('''
!python backtest.py
''', stdout="""
  period                 1990-07-03 -> 2024-03-28
  trading days           8,565
  markets                34 (live/day: median 31, max 34)
  annualised return       11.91%
  annualised volatility   15.54%   (target 20%)
  Sharpe (excess=0)         0.77
  max drawdown           -33.51%
  monthly skewness          1.04   (trend following should be positive)
  share of days short     37.50%

  Sharpe by decade:
    1990s  ann +11.64%  vol 13.98%  Sharpe +0.83
    2000s  ann +18.26%  vol 18.02%  Sharpe +1.01
    2010s  ann  +4.11%  vol 13.85%  Sharpe +0.30
    2020s  ann +16.30%  vol 16.38%  Sharpe +1.00
""")

fig("val_fig1_equity.png", "The rebuilt backtest, 1990–2024")

md("""
### Is it credible? Four checks, not decorations

- **Monthly skewness +1.04.** Positive skew is trend following's structural signature — many
  small losses, occasional large gains. A backtest without it is not trend following.
- **Sharpe by decade 0.83 / 1.01 / 0.30 / 1.00.** The 2010s trough and 2020s recovery match the
  documented industry pattern. A backtest that performed *uniformly* well would be the
  suspicious one.
- **37.5% of market-days short.** Genuinely two-sided — the property our own data could not give.
- **Max drawdown −33.5%** against a published 12-market replication's −30%, with a higher Sharpe
  consistent with 34 markets rather than 12.

**Gross of transaction costs**, which inflates the Sharpe ratio. It does not materially affect
the tail behaviour under test.
""")

md("""
---

## Part III — The coverage test

Two models, scored walk-forward on identical days with identical positions:

- **OLD** — multivariate Student-t on an EWMA covariance matrix, one shared tail parameter for
  all markets, no mechanism for volatility clustering.
- **NEW** — filtered historical simulation: per-market GARCH(1,1) for conditional volatility,
  then whole-day vectors of standardised residuals resampled from the trailing pool, so each
  market keeps its own tail shape and cross-market structure survives without a correlation
  model.

Parameters refitted annually on trailing data; between refits the volatility recursion runs
forward on realised returns, which is a one-step-ahead filter rather than a smoother. The
residual pool at date *t* contains only residuals dated before *t*.

### A bug found before the result

The backtest's trading index has gaps, so recovering "yesterday's position" by shifting one row
picks a *non-adjacent* day wherever a gap falls — 95 of 8,564 rows. Fixed by lagging the position
and volatility panels **before** filtering and saving them that way, so nothing downstream
re-derives them. The test now carries an identity check: exposure times market moves must
reproduce the realised loss. It does, to 1.4 × 10⁻¹⁴.

The fix moved the 99% breach rate from 2.55% to 2.49% — which is the only reason the failure
below is worth reporting. It is the model's, not the plumbing's.
""")

code('''
!python coverage_clean.py
''', stdout="""
COVERAGE TEST - clean long/short backtest, 34 markets
  test days 7,305   1995-06-27 -> 2024-03-28

  model    level  target  actual  breaches   Kupiec   indep  ES ratio  verdict
  --------------------------------------------------------------------------
  OLD        90%   10.0%  12.03%       879    0.000   0.003      1.11  FAILS
  OLD        95%    5.0%   7.19%       525    0.000   0.214      1.10  FAILS
  OLD        99%    1.0%   2.34%       171    0.000   0.340      1.07  FAILS
  NEW        90%   10.0%  12.81%       936    0.000   0.000      1.14  FAILS
  NEW        95%    5.0%   7.86%       574    0.000   0.000      1.10  FAILS
  NEW        99%    1.0%   2.49%       182    0.000   0.270      1.10  FAILS

  Tail shape, conditional (standardised by each model's own 90%):
    OLD   realised 2.30   modelled 2.03
    NEW   realised 2.18   modelled 2.10
""")

md("""
### Both fail — and tail shape is not why

Look at the shape row. The NEW model's conditional tail shape is 2.10 against a realised 2.18 —
about 5% flat, which is the magnitude Part I predicted for a filtered-historical method. Shape is
close to right. Something else is wrong.

**Two measurements that would have misled us, named here because both are easy to reach for:**

1. *Backing a portfolio σ out of the model's own 90% quantile as VaR₉₀ / 1.2816.* The FHS
   distribution is fat-tailed, so its 90% quantile is not 1.2816σ. Dividing by 1.2816 understates
   σ and manufactures a scale shortfall that is an artefact of assuming normality. This produced
   an apparent 29% gap that does not exist.
2. *Comparing the pooled realised quantile against the mean predicted quantile.* Pooling realised
   losses across days of different volatility mixes an unconditional object with a conditional
   one. On a synthetic process where both are known exactly, the pooled ratio is 2.55 against a
   conditional 2.18 — a **1.17× spread present by construction, with the model exactly right**.
   The real data shows 1.22×. A comparison that ignores this reports a 46% tail deficit where
   there is none.

What is used instead: breach rates, which are conditional by construction; the scale-free ratio
loss / VaR₉₀ for *timing only*; and a same-day comparison of the model's scale against a trailing
realised covariance on identical exposures, which compares two conditional objects.
""")

code('''
!python diagnose.py
''', stdout="""
1. IS THE MISS A CONSTANT, OR IS IT TIMED?
  autocorrelation of |loss / VaR90|:
    lag 1  +0.046    lag 5  +0.047    lag 20  +0.025
  -> close to nothing. Not a dynamic effect the model fails to track.

2. HOW MUCH SCALE IS MISSING, MEASURED HONESTLY
    trailing covariance window    60d:   realised / model = 1.105
    trailing covariance window   250d:   realised / model = 1.096
    trailing covariance window  1000d:   realised / model = 1.081

  five largest standardised losses:
    2003-03-13  6.83    2015-01-16  6.74    2018-02-05  6.42
    2022-03-09  5.87    2022-11-10  5.61
""")

md("""
### The diagnosis

The miss is **scale, and it is persistent rather than timed.** The autocorrelation of
|loss / VaR₉₀| is +0.046 at lag 1 — almost no dynamic structure is left for a smarter volatility
model to capture. Meanwhile the same-day exposures run through a trailing realised covariance
give 8–11% more risk than the model's own number, almost all of the time.

Since each market's own volatility is already filtered by its own GARCH, **what is missing is a
portfolio-level quantity: how much the markets move together.** A trend book is concentrated in
exactly that — long the whole bond complex, or short the whole dollar bloc, at once — so the
portfolio is less diversified than per-market risk accounting implies.

This is the mechanism the plan of record calls correlation convergence. It is real. It is just
not shaped like a regime switch.
""")

md("""
---

## Part IV — The correction, chosen blind

### Why the split comes first

The obvious way to pick a correction is to try several and keep whichever gives the best
full-sample coverage. **That is selection on the test set.** With a handful of candidates and
three confidence levels, there is enough freedom to land near target by luck alone, and the
result would mean nothing.

So the sample splits at **2010-01-01**. Every choice — the form of the correction and its window
— is made on the training half. The number reported is the held-out half.

Two candidates, both strictly trailing:

- **constant** — one number *k*, the 90th percentile of loss / VaR₉₀ on the training half, then
  frozen. No adaptation at all. This is the baseline the adaptive version must beat.
- **trailing** — *k* re-estimated daily as the 90th percentile of loss / VaR₉₀ over a trailing
  window.
""")

code('''
!python portfolio_layer.py
''', stdout="""
  constant correction estimated on training: x1.253
  (the model's VaR is 25% too small)

  Trailing-window selection, TRAINING half only:
    window     90%     95%     99%    error
       250   10.49    5.52    1.58    1.039
       500   10.38    5.52    1.45    0.615  <- best
      1000   10.33    5.57    1.53    0.857
      2000   10.55    5.55    1.50    0.773

ALL THREE LEVELS, HELD-OUT HALF - Kupiec p for every candidate
  correction              90%      p    95%      p    99%      p   verdict
  ----------------------------------------------------------------------
  uncorrected           11.96  0.000   6.94  0.000   2.17  0.000   fails
  constant x1.253        8.26  0.000   3.98  0.003   0.93  0.680   fails
  trailing 250d         10.21  0.680   5.62  0.090   1.40  0.022   fails
  trailing 500d         10.34  0.492   5.35  0.338   1.37  0.033   fails  *
  trailing 1000d         9.99  0.978   5.02  0.955   1.29  0.093   PASSES all three
  trailing 2000d         9.66  0.488   4.80  0.579   1.26  0.127   PASSES all three

  * = the pre-registered choice
""")

fig("val_fig2_coverage.png", "Coverage on the held-out half, before and after correction")

md("""
### Reading this honestly

**The model's VaR is 25% too small.** That is the headline, estimated on training data alone.

**The pre-registered choice — 500 days — gets the body right and the tail marginally wrong** out
of sample: 10.34% and 5.35% against targets of 10% and 5%, but 1.37% against 1%, which Kupiec
rejects at p = 0.033. That is the primary result, because it is what the selection procedure
actually chose.

**Longer windows do better.** A 1,000-day window gives 9.99 / 5.02 / 1.29 and passes at all three
levels; 2,000 days also passes. Note the direction: the training half selected a window that is
*not* the best out-of-sample performer. **The selection worked against the reported result rather
than flattering it**, which is the right direction for that kind of error to run.

**The constant baseline is informative in its failure.** Frozen at ×1.253 it passes at 99%
(p = 0.680) but becomes too conservative in the body — 8.26% and 3.98%. So adaptivity buys
something: the trailing version holds all three levels where the constant holds one. The gap is
modest, and longer windows converge toward the constant, consistent with the shortfall being a
stable feature rather than a moving one.

**Recommendation: adopt the 1,000-day trailing window**, and report the 500-day result alongside
it so the selection history is visible.
""")

fig("val_fig3_rolling.png",
    "The same result through time — rolling 500-day breach rate at the 99% level")

md("""
---

## Does the synthetic half predict the real half?

This is the test that decides whether Part I was worth doing. Take the parameters measured on
real filtered residuals — tail index 6.9, skewness −0.21, persistence 0.984 — find the matching
cell in the synthetic grid, and read off what the study says coverage should be.

The conditional-versus-pooled spread that Part III depends on transfers almost exactly:

| | conditional | pooled | spread |
|---|---|---|---|
| synthetic process, both known exactly | 2.18 | 2.55 | 1.17× |
| real data, 1,972 matched observations | 2.20 | 2.54 | 1.22× |

The synthetic study predicts the real data's behaviour to within a few percent on the quantity
that matters. Its other conclusions can be relied on.
""")

code('''
!python transfer_test.py
''', stdout="""
TAIL SHAPE  (VaR99 / VaR90)
  TRUE shape ratio for this process: 2.18

  method            250     500    1000    2000
  naive hist       2.29    2.35    2.43    2.48
  moving avg       1.82    1.82    1.82    1.82
  EWMA             1.82    1.82    1.82    1.82
  GARCH-N          1.82    1.82    1.82    1.82
  GARCH-t          2.08    2.10    2.11    2.11

  A normal quantile forces the ratio to 1.82 regardless of the data.

CONDITIONAL vs UNCONDITIONAL
  SYNTHETIC, both known exactly:   conditional 2.18   pooled 2.55
  REAL, 1,972 matched obs:         conditional 2.20   pooled 2.54
  spread:  synthetic 1.17x,  real 1.22x
""")

md("""
---

## Limitations

Stated at the level a reviewer will press on.

**The strategy is a reimplementation, not the production signal.** The rule follows the published
book our strategy is based on and correlates 0.964 with observed production signals, but it is
not the same code. Conclusions about *estimator behaviour* transfer; exact risk numbers for the
live book do not.

**No transaction costs**, so the backtest's Sharpe is optimistic. Tail behaviour under test is
unaffected.

**Two markets missing** of 36, and the market set drifts — 24 live at the start of the test
period, 34 at the end.

**One split, not many.** The held-out half is a single contiguous period, 2010–2024, and the
2010s were an unusual decade for this strategy. Rolling-origin validation would be stronger.

**Only the one-day horizon is tested here.** The coverage test scores a one-day horizon, where
the book does not re-trade, so nothing in this study exercises the engine's path-dependent
position simulation or speaks to whipsaw. That is a property of this test, not a missing piece of
the engine — the engine does re-run the trend rule along each simulated path.

**Monthly and annual horizons cannot be validated with available data.** 167 independent monthly
observations give a 99% test no power: it would accept any true breach rate between 0% and 2.4%.
""")

md("""
---

## What follows

**Immediately actionable**

1. Wire the filtered-historical generator into the multi-day path loop. It currently feeds only
   the one-day path; the multi-day loop still draws from the older distribution. Contained change.
2. Add the scale correction as a parameter on the engine's output.
3. Make the 99/90 shape ratio a standing diagnostic. It is scale-free, separates shape error from
   level error, and instantly identifies any estimator reading its quantile off a normal.

**Needs a decision** — the five items in the table near the top.

**Blocked on the database** — validation against our own strategy's realised P&L, which is the
only thing that can confirm these numbers for the live book rather than for a faithful
reimplementation of it.

### Reproducing this

```bash
cd code
python validate_steps123.py     # 15 checks on the synthetic machinery
python analyse_grid.py          # synthetic figures + crossover tables
python transfer_test.py         # synthetic-to-real transfer test
python backtest.py              # rebuild the backtest from public prices
python coverage_clean.py        # walk-forward coverage test (~5 min)
python diagnose.py              # why it fails
python portfolio_layer.py       # the correction, train/test split
python make_figs.py             # real-data figures
```

### Sources

- Prices and reference implementation: [pysystemtrade](https://github.com/robcarver17/pysystemtrade)
- Benchmark replication: [cross-asset-trend-following](https://github.com/marcinossolinski/cross-asset-trend-following)
- Filtered historical simulation: Barone-Adesi, Giannopoulos & Vosper (1999)
- Skewed-t density: Hansen (1994)
- Coverage tests: Kupiec (1995), Christoffersen (1998)
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "research_notebook.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print(f"wrote {out} ({len(cells)} cells)")
