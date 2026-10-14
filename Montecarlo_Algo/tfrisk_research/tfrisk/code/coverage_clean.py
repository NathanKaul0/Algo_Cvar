"""
The coverage test, on the clean backtest. Phase 1's stated pass/fail criterion.

WHAT CHANGED FROM THE PREVIOUS RUN
----------------------------------
Everything except the models. The previous coverage test ran on the club's
backtest database and survived a quality screen that left 1,972 matched
observations from one effectively long-only run. This runs on the rebuilt
public-price backtest: 8,565 days, 34 markets, 37.5% of market-days short. That
is the first time either model has been scored on a genuinely long/short book,
which the Phase 1 document names as a requirement and which the old substrate
could not supply.

THE TWO MODELS
--------------
  OLD   multivariate Student-t with an EWMA covariance matrix - one shared
        tail parameter for all markets, no volatility clustering mechanism.
  NEW   filtered historical simulation - per-market GARCH(1,1) for the
        conditional volatility, then whole-day vectors of standardised
        residuals resampled from the trailing pool, so each market keeps its
        own tail shape and the cross-market structure is preserved without a
        correlation model.

UNITS
-----
Back-adjusted prices can cross zero, so each market's daily move is expressed
as x = delta_price / trailing_vol, a scale-free quantity with roughly unit
variance, and the exposure vector carries the units back:

    exposure_i = contracts_i * trailing_vol_i      (decided at t-1)
    loss_t     = -sum_i exposure_i * x_i,t

NO LOOK-AHEAD
-------------
Parameters are refitted every REFIT days on trailing data only; between refits
the volatility recursion is run forward using realised returns, which is a
one-step-ahead filter rather than a smoother. The residual pool at date t
contains only residuals dated before t. Positions come from the backtest and
were decided at t-1.

    python coverage_clean.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # var_estimators lives alongside this file
from var_estimators import fit_garch_forecast                   # noqa: E402
from paths import data_path, fig_path  # noqa: E402

LEVELS = (0.90, 0.95, 0.99)
BURN = 1260          # 5 years before the first test date
REFIT = 250          # refit parameters once a year
POOL = 1000          # residual pool length for FHS
N_SIM = 20_000
EWMA_LAM = 0.94
VOL_SPAN = 36
ANN = 256.0


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load() -> dict:
    px = pd.read_csv(data_path("prices_daily_pst.csv"),
                     index_col=0, parse_dates=True).sort_index()
    # the LAGGED panels, saved by the backtest already aligned to the day
    # whose P&L they earn. Not re-derived by shifting here: the index has gaps,
    # so a shift would pick non-adjacent days.
    pos = pd.read_csv(data_path("positions_effective.csv"),
                      index_col=0, parse_dates=True)
    vol_eff = pd.read_csv(data_path("vol_effective.csv"),
                          index_col=0, parse_dates=True)
    ret = pd.read_csv(data_path("strategy_returns.csv"),
                      index_col=0, parse_dates=True)["ret"]

    dp = px.diff()
    vol_d = dp.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std()
    x = (dp / vol_d.shift(1))                       # scale-free daily move
    x = x.replace([np.inf, -np.inf], np.nan)

    idx = ret.index.intersection(x.index).intersection(pos.index)
    return {"x": x.loc[idx], "vol_d": vol_eff.loc[idx], "pos": pos.loc[idx],
            "ret": ret.loc[idx], "idx": idx}


# ---------------------------------------------------------------------------
# the two models
# ---------------------------------------------------------------------------

def garch_filter(x: np.ndarray, a: float, b: float, w: float) -> np.ndarray:
    """One-step-ahead conditional variance path. Uses only past returns.

    v[t] is the variance forecast FOR day t, built from returns through t-1, so
    v has one more element than x and v[-1] is tomorrow's forecast. Written as
    a linear filter rather than a Python loop: the recursion is an AR(1) in the
    variance driven by the squared return, and lfilter evaluates it in C. The
    loop version was the bottleneck - it is O(n) per day and was being re-run
    every day, which is O(n^2) over the walk.
    """
    from scipy.signal import lfilter
    v0 = w / max(1e-12, 1 - a - b)
    drive = np.empty(len(x) + 1)
    drive[0] = v0 * (1 - b)                # so the recursion starts at v0
    drive[1:] = w + a * np.asarray(x, dtype=float) ** 2
    return lfilter([1.0], [1.0, -b], drive)


def fit_panel(xw: pd.DataFrame) -> dict:
    """Fit per-market GARCH(1,1) on a trailing window."""
    par = {}
    for c in xw.columns:
        s = xw[c].dropna().to_numpy()
        if len(s) < 300:
            continue
        f = fit_garch_forecast(s, dist="normal")
        if not f.get("ok", 1.0):
            continue
        a, b = f["alpha"], f["beta"]
        # the fitter uses variance targeting, so omega is implied by the
        # window's own variance rather than estimated freely
        par[c] = (a, b, float(np.var(s)) * max(1e-6, 1.0 - a - b))
    return par


def fhs_quantiles(sig_next: np.ndarray, pool: np.ndarray, expo: np.ndarray,
                  rng) -> dict:
    """Resample whole days of standardised residuals, scale, apply exposures.

    Drawing whole rows rather than per-market values is what preserves the
    cross-market structure: a day on which eight markets all moved against the
    book stays a single draw, and no correlation matrix is involved.
    """
    k = len(expo)
    draw = pool[rng.integers(0, len(pool), N_SIM)]       # (N_SIM, k)
    sim = draw * sig_next[None, :]
    pnl = sim @ expo
    loss = -np.sort(pnl)[::-1]
    loss = np.sort(loss)
    out = {}
    for q in LEVELS:
        v = float(np.quantile(loss, q))
        out[q] = (v, float(loss[loss >= v].mean()))
    return out


def student_t_quantiles(xw: pd.DataFrame, expo: np.ndarray, rng,
                        nu_cache: dict | None = None) -> dict:
    """OLD model: multivariate Student-t on an EWMA covariance matrix."""
    X = xw.to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    n, k = X.shape
    wts = EWMA_LAM ** np.arange(n - 1, -1, -1)
    wts /= wts.sum()
    Xc = X - (wts[:, None] * X).sum(axis=0)
    S = (Xc * wts[:, None]).T @ Xc
    S += np.eye(k) * 1e-10

    # one shared tail parameter, fitted on the pooled standardised series.
    # Refitted only at the walk's refit boundaries, not daily: it is a slow-
    # moving quantity and the MLE is the single most expensive call here.
    if nu_cache is not None and "nu" in nu_cache:
        nu = nu_cache["nu"]
    else:
        sd = np.sqrt(np.clip(np.diag(S), 1e-16, None))
        z = (Xc / sd).ravel()
        z = z[np.isfinite(z)]
        nu = float(np.clip(sps.t.fit(z, floc=0, fscale=1)[0], 3.0, 30.0))
        if nu_cache is not None:
            nu_cache["nu"] = nu

    L = np.linalg.cholesky(S)
    g = rng.standard_normal((N_SIM, k))
    chi = rng.chisquare(nu, N_SIM)
    sim = (g @ L.T) * np.sqrt(nu / chi)[:, None] * np.sqrt((nu - 2) / nu)
    loss = np.sort(-(sim @ expo))
    out = {}
    for q in LEVELS:
        v = float(np.quantile(loss, q))
        out[q] = (v, float(loss[loss >= v].mean()))
    return out


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------

def walk_forward(d: dict, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x, vol_d, pos, ret, idx = d["x"], d["vol_d"], d["pos"], d["ret"], d["idx"]
    cols = list(x.columns)

    X = x.to_numpy(dtype=float)
    Xf = np.where(np.isfinite(X), X, 0.0)
    P = pos.reindex(columns=cols).to_numpy(dtype=float)
    V = vol_d.reindex(columns=cols).to_numpy(dtype=float)
    EXPO = np.where(np.isfinite(P * V), P * V, 0.0)
    LOSS = -ret.to_numpy(dtype=float)

    rows, pool_cols, SIG, keep_ix, nu_cache = [], [], None, None, {}
    print(f"  walking {len(idx) - BURN:,} test days, refit every {REFIT}...")
    for i in range(BURN, len(idx)):
        t = idx[i]

        if (i - BURN) % REFIT == 0:
            par = fit_panel(x.iloc[max(0, i - POOL * 2):i])
            pool_cols = [c for c in cols if c in par]
            keep_ix = np.array([cols.index(c) for c in pool_cols], dtype=int)
            # one filter pass over the WHOLE series per market, with this
            # refit's parameters; v[j] uses returns through j-1 only, so
            # reading v[i] on day i stays one-step-ahead
            SIG = np.column_stack([
                np.sqrt(np.clip(garch_filter(Xf[:, cols.index(c)], *par[c]),
                                1e-18, None))[:len(idx)]
                for c in pool_cols]) if pool_cols else None
            nu_cache.clear()
            print(f"    {t.date()}  refit {len(par)} markets"
                  f"   [{i - BURN:,}/{len(idx) - BURN:,}]", flush=True)

        if not pool_cols:
            continue

        expo = EXPO[i, keep_ix]                # already lagged by the backtest
        if not np.any(expo):
            continue

        sig_next = SIG[i, :]                   # forecast FOR day i

        # residual pool: strictly before t
        lo = max(0, i - POOL)
        Z = X[lo:i][:, keep_ix] / SIG[lo:i, :]
        ok = np.isfinite(Z).all(axis=1)
        Z = Z[ok]
        if len(Z) < 250:
            continue

        new = fhs_quantiles(sig_next, Z, expo, rng)
        old = student_t_quantiles(x[pool_cols].iloc[lo:i], expo, rng, nu_cache)

        row = {"date": t, "loss": float(LOSS[i])}
        for q in LEVELS:
            row[f"new_var_{q}"], row[f"new_cvar_{q}"] = new[q]
            row[f"old_var_{q}"], row[f"old_cvar_{q}"] = old[q]
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def kupiec(n: int, k: int, p: float) -> float:
    if k == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        ph = k / n
        lr = -2 * ((n - k) * np.log((1 - p) / (1 - ph)) + k * np.log(p / ph))
    return float(1 - sps.chi2.cdf(lr, 1))


def christoffersen(br: np.ndarray) -> float:
    """Independence test: are breaches clustered?"""
    a = br[:-1].astype(int) * 2 + br[1:].astype(int)
    n00, n01, n10, n11 = [(a == v).sum() for v in (0, 1, 2, 3)]
    if (n00 + n01) == 0 or (n10 + n11) == 0 or (n01 + n11) == 0:
        return float("nan")
    p01, p11 = n01 / (n00 + n01), n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    def ll(x, k, m):
        return 0.0 if x <= 0 or x >= 1 else k * np.log(x) + m * np.log(1 - x)
    lr = -2 * (ll(p, n01 + n11, n00 + n10)
               - ll(p01, n01, n00) - ll(p11, n11, n10))
    return float(1 - sps.chi2.cdf(lr, 1))


def score(r: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("COVERAGE TEST - clean long/short backtest, 34 markets")
    print("=" * 78)
    print(f"  test days {len(r):,}   "
          f"{r.index.min().date()} -> {r.index.max().date()}")
    print(f"\n  {'model':<7}{'level':>7}{'target':>8}{'actual':>8}"
          f"{'breaches':>10}{'Kupiec':>9}{'indep':>8}{'ES ratio':>10}  verdict")
    print("  " + "-" * 74)
    for tag, nm in (("old", "OLD"), ("new", "NEW")):
        for q in LEVELS:
            br = (r.loss > r[f"{tag}_var_{q}"]).to_numpy()
            k, n = int(br.sum()), len(r)
            rate, p = k / n, 1 - q
            ku, ind = kupiec(n, k, p), christoffersen(br)
            # realised tail mean vs predicted, on the breach days
            es = (r.loss[br].mean() / r[f"{tag}_cvar_{q}"][br].mean()
                  if k > 2 else float("nan"))
            tag_v = "PASS" if ku > 0.05 else ("marginal" if ku > 0.01 else "FAILS")
            print(f"  {nm:<7}{q:>7.0%}{p:>8.1%}{rate:>8.2%}{k:>10,}"
                  f"{ku:>9.3f}{ind:>8.3f}{es:>10.2f}  {tag_v}")

    print("\n  ES ratio is realised tail loss over predicted, on breach days:")
    print("  above 1 means the breaches were worse than the model said they")
    print("  would be, which is the number CVaR exists to get right.")

    print("\n  Tail shape, conditional (standardised by each model's own 90%):")
    for tag, nm in (("old", "OLD"), ("new", "NEW")):
        z = (r.loss / r[f"{tag}_var_0.9"]).to_numpy()
        z = z[np.isfinite(z)]
        print(f"    {nm:<5} realised {np.quantile(z,0.99)/np.quantile(z,0.90):.2f}"
              f"   modelled "
              f"{(r[f'{tag}_var_0.99']/r[f'{tag}_var_0.9']).mean():.2f}")


if __name__ == "__main__":
    d = load()
    print(f"loaded {len(d['idx']):,} days, {d['x'].shape[1]} markets")
    r = walk_forward(d)
    r.to_csv(data_path("coverage_clean_results.csv"))
    score(r)
    print(f"\n  wrote coverage_clean_results.csv ({len(r):,} rows)")
