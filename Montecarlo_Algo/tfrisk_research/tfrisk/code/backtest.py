"""
A clean trend-following backtest on public data, to replace the corrupted one.

WHY THIS EXISTS
---------------
The coverage test is Phase 1's stated pass/fail criterion, and it needs a
backtest to score against. The club's backtest database cannot serve: it is a
collection of unverified runs, and the quality screen left 18 of 549 usable and
exactly one holding short positions. So the substrate is rebuilt from public
prices instead, with a construction that is fully known because we wrote it.

Prices: pysystemtrade (github.com/robcarver17/pysystemtrade), Rob Carver's own
repository - the same author as the book this strategy follows. 34 of the
strategy's 36 markets are present (Brazilian real and Kansas wheat are not).
Daily closes resampled from intraday bars, 1990-01 to 2024-03.

THE RULE
--------
The same reimplemented Carver EWMAC rule used everywhere else in Phase 1:
three speed pairs (4/16, 16/64, 32/128), Carver's forecast scalars, a hard
+-20 cap per speed and on the combined forecast, FDM 1.45, 20% annualised
volatility target, 10% position buffer.

WHY PRICE POINTS RATHER THAN PERCENT RETURNS
--------------------------------------------
Back-adjusted continuous futures prices can pass through zero or go negative
(2020 crude is the obvious case), so percentage returns on them are not
meaningful. Everything below therefore works in price differences normalised by
a volatility measured in the same price units, which is also how Carver frames
it. One consequence is worth stating because it removes a whole class of
objection: with

    contracts_i = (forecast_i / 10) * (capital * vol_target / N) / vol_annual_i
    pnl_i       = contracts_i * delta_price_i

the price units cancel exactly, so the contract multiplier and the quote
currency drop out of the risk-targeted P&L. Missing contract specs and FX rates
cannot bias this backtest.

NO LOOK-AHEAD
-------------
Every forecast and volatility estimate at date t uses data through t-1 only,
and the position decided at t-1 earns the P&L of t. The shift is explicit.

    python backtest.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from paths import data_path, fig_path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

PAIRS = ((4, 16), (16, 64), (32, 128))
SCALARS = {4: 4.10, 16: 2.79, 32: 1.91}      # Carver's, by fast span
CAP = 20.0
FDM = 1.45
VOL_TARGET = 0.20
BUFFER = 0.10
VOL_SPAN = 36
ANN = 256.0
MIN_LIVE = 10                                 # don't trade a 3-market book


def load_prices() -> pd.DataFrame:
    px = pd.read_csv(data_path("prices_daily_pst.csv"),
                     index_col=0, parse_dates=True)
    return px.sort_index()


def forecasts(px: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combined capped forecast and annualised price-point volatility.

    Both are lagged one day before use by the caller, so nothing here needs to
    know about the shift.
    """
    dp = px.diff()
    vol_d = dp.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std()
    vol_a = vol_d * np.sqrt(ANN)

    fc = None
    for fast, slow in PAIRS:
        raw = px.ewm(span=fast, min_periods=fast).mean() - \
              px.ewm(span=slow, min_periods=slow).mean()
        scaled = (raw / vol_d) * SCALARS[fast]
        capped = scaled.clip(-CAP, CAP)
        fc = capped if fc is None else fc + capped
    fc = (fc / len(PAIRS) * FDM).clip(-CAP, CAP)
    return fc, vol_a


def apply_buffer(target: pd.DataFrame) -> pd.DataFrame:
    """Hold the current position unless the target has drifted past the buffer.

    Carver's buffer is in contracts; here it is a fraction of the target's own
    scale, which is the same idea in normalised units. Without it the book
    re-trades every market every day, which would overstate turnover and
    understate the position staleness that risk #4 is about.
    """
    tgt = target.to_numpy(dtype=float)
    held = np.zeros(tgt.shape[1])
    out = np.empty_like(tgt)
    scale = np.nanstd(tgt, axis=0)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    tol = BUFFER * scale
    for i in range(tgt.shape[0]):
        row = tgt[i]
        live = np.isfinite(row)
        move = live & (np.abs(row - held) > tol)
        held = np.where(move, row, held)
        held = np.where(live, held, 0.0)       # exit a market that goes dark
        out[i] = held
    return pd.DataFrame(out, index=target.index, columns=target.columns)


def idm_series(px: pd.DataFrame, vol_a: pd.DataFrame, live: pd.DataFrame,
               step: int = 63, window: int = 504, cap: float = 2.5) -> pd.Series:
    """Instrument diversification multiplier, estimated on trailing data only.

    Equal risk weights across N markets do not produce N times the risk of one
    market, because the markets are correlated: the book lands far below its
    volatility target. Carver's fix is to scale the whole book by
    1 / sqrt(w' C w), capped - the ratio between the risk an uncorrelated book
    would carry and what this correlation structure actually gives.

    Re-estimated quarterly on the trailing two years and held constant in
    between, so it is a trailing quantity throughout. Without it the backtest
    runs at roughly a third of its stated risk target, which would make every
    loss number quietly too small.
    """
    z = (px.diff() / (vol_a / np.sqrt(ANN))).where(live)       # risk-normalised
    idx = px.index
    vals = pd.Series(np.nan, index=idx, dtype=float)
    for i in range(window, len(idx), step):
        blk = z.iloc[i - window:i]
        cols = blk.columns[blk.notna().sum() >= window * 0.8]
        if len(cols) < MIN_LIVE:
            continue
        C = blk[cols].corr().to_numpy()
        C = np.nan_to_num(C, nan=0.0)
        np.fill_diagonal(C, 1.0)
        w = np.full(len(cols), 1.0 / len(cols))
        v = float(w @ C @ w)
        vals.iloc[i] = min(cap, 1.0 / np.sqrt(v)) if v > 0 else np.nan
    return vals.ffill()


def run() -> dict:
    px = load_prices()
    fc, vol_a = forecasts(px)
    dp = px.diff()

    live = px.notna() & fc.notna() & vol_a.notna() & (vol_a > 0)
    n_live = live.sum(axis=1)
    idm = idm_series(px, vol_a, live)

    # risk budget per market, equal-weighted across whatever is live that day,
    # then scaled up by the diversification multiplier
    budget = VOL_TARGET / n_live.replace(0, np.nan) * idm
    target = (fc / 10.0).where(live) * budget.to_numpy()[:, None] / vol_a
    held = apply_buffer(target)

    # position decided at t-1 earns the P&L of t
    held_eff = held.shift(1)                 # the position that EARNS day t
    vol_eff = vol_a.shift(1) / np.sqrt(ANN)  # the vol it was sized on, daily
    pnl_mkt = held_eff * dp
    ret = pnl_mkt.sum(axis=1, min_count=1)

    keep = (n_live >= MIN_LIVE) & ret.notna()

    # The effective panels are lagged BEFORE filtering and saved that way.
    # Anything downstream that re-derives them by shifting the filtered index
    # gets it wrong, because the filtered index has gaps: a one-row shift then
    # picks a non-adjacent day. That cost 95 misaligned rows the first time.
    out = {"ret": ret[keep], "pnl_mkt": pnl_mkt[keep], "pos": held[keep],
           "pos_eff": held_eff[keep], "vol_eff": vol_eff[keep],
           "n_live": n_live[keep], "idm": idm[keep]}
    return out


def report(res: dict) -> None:
    r = res["ret"]
    ann_ret, ann_vol = r.mean() * ANN, r.std() * np.sqrt(ANN)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    from scipy import stats as sps

    print("=" * 74)
    print("CLEAN BACKTEST - reimplemented EWMAC on public pysystemtrade prices")
    print("=" * 74)
    print(f"  period                 {r.index.min().date()} -> {r.index.max().date()}")
    print(f"  trading days           {len(r):,}")
    print(f"  markets                {res['pos'].shape[1]} "
          f"(live/day: median {int(res['n_live'].median())}, "
          f"max {int(res['n_live'].max())})")
    print(f"  annualised return      {ann_ret:7.2%}")
    print(f"  annualised volatility  {ann_vol:7.2%}   (target {VOL_TARGET:.0%})")
    print(f"  Sharpe (excess=0)      {ann_ret / ann_vol:7.2f}")
    print(f"  max drawdown           {dd:7.2%}")
    print(f"  daily skewness         {sps.skew(r):7.2f}")
    print(f"  daily excess kurtosis  {sps.kurtosis(r):7.2f}")
    m = r.resample("ME").sum()
    print(f"  monthly skewness       {sps.skew(m):7.2f}   "
          f"(trend following should be positive)")
    print(f"  share of days short    "
          f"{(res['pos'] < 0).sum().sum() / res['pos'].notna().sum().sum():7.2%}"
          f"   <- the corrupted data had essentially none")

    print("\n  Sharpe by decade (sanity: trend decays after 2010):")
    for dec, g in r.groupby(r.index.year // 10 * 10):
        if len(g) < 250:
            continue
        print(f"    {dec}s  n={len(g):5d}  "
              f"ann {g.mean()*ANN:+7.2%}  vol {g.std()*np.sqrt(ANN):6.2%}  "
              f"Sharpe {g.mean()*ANN/(g.std()*np.sqrt(ANN)):+5.2f}")


if __name__ == "__main__":
    res = run()
    report(res)
    res["ret"].to_frame("ret").to_csv(data_path("strategy_returns.csv"))
    res["pos"].to_csv(data_path("positions.csv"))
    res["pos_eff"].to_csv(data_path("positions_effective.csv"))
    res["vol_eff"].to_csv(data_path("vol_effective.csv"))
    res["pnl_mkt"].to_csv(data_path("pnl_by_market.csv"))
    print("\n  wrote strategy_returns.csv, positions.csv, "
          "positions_effective.csv, vol_effective.csv, pnl_by_market.csv")
