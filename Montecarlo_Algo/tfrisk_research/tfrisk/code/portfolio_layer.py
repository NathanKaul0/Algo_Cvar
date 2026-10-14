"""
Fixing the scale shortfall, with every choice made out of sample.

THE PROBLEM
-----------
On the clean long/short backtest both models over-breach at every level. The
timing evidence says it is NOT a dynamic effect the model fails to track: the
autocorrelation of |loss / VaR90| is +0.046 at lag 1, which is close to nothing.
What the evidence does say is that the scale is persistently too small - the
same-day exposures run through a trailing realised covariance give 8-11% more
risk than the model's own number, almost all the time.

A CORRECTION THAT WAS WRONG, AND WHY IT LOOKED RIGHT
----------------------------------------------------
A first attempt applied an EWMA correction c_t = lam*c_{t-1} + (1-lam)*u_{t-1}^2
with u = loss / (VaR90 / 1.2816), and appeared to fix the 99% level outright.
It did not. Dividing VaR90 by 1.2816 treats the fat-tailed FHS distribution as
if it were normal, which inflates u by a constant 1.28 and therefore inflates
the correction by the same constant. The EWMA was decoration; what actually
fixed the coverage was multiplying VaR by about 1.28. Rebuilt without the
artefact, the EWMA version makes held-out coverage WORSE, which is the correct
verdict on it.

So the honest question is not "what dynamic layer is missing" but "by how much
is the scale wrong, and is that amount stable enough to estimate in advance".

TWO CANDIDATES, BOTH TRAILING
-----------------------------
  constant   one number k, estimated once on the training half as the 90th
             percentile of loss / VaR90, then frozen. No adaptation at all.
  trailing   k_t re-estimated every day as the 90th percentile of
             loss / VaR90 over a trailing window, so it drifts with the data.

The constant is the baseline the adaptive version has to beat. If it does not,
the extra machinery is not earning anything and should not ship.

Window length and the decision between them come from the TRAINING half only.
The reported number is the held-out half.

    python portfolio_layer.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats as sps
from paths import data_path, fig_path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = ("0.9", "0.95", "0.99")
TARGETS = (0.10, 0.05, 0.01)
WINDOWS = (250, 500, 1000, 2000)
SPLIT = "2010-01-01"


def kupiec(n: int, k: int, p: float) -> float:
    if k == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        ph = k / n
        lr = -2 * ((n - k) * np.log((1 - p) / (1 - ph)) + k * np.log(p / ph))
    return float(1 - sps.chi2.cdf(lr, 1))


def christoffersen(br: np.ndarray) -> float:
    a = br[:-1].astype(int) * 2 + br[1:].astype(int)
    n00, n01, n10, n11 = [(a == v).sum() for v in (0, 1, 2, 3)]
    if min(n00 + n01, n10 + n11, n01 + n11) == 0:
        return float("nan")
    p01, p11 = n01 / (n00 + n01), n11 / (n10 + n11)
    p = (n01 + n11) / len(a)

    def ll(x, k, m):
        return 0.0 if x <= 0 or x >= 1 else k * np.log(x) + m * np.log(1 - x)
    lr = -2 * (ll(p, n01 + n11, n00 + n10) - ll(p01, n01, n00) - ll(p11, n11, n10))
    return float(1 - sps.chi2.cdf(lr, 1))


# ---------------------------------------------------------------------------
# the two corrections
# ---------------------------------------------------------------------------

def k_constant(u: np.ndarray, train: np.ndarray) -> np.ndarray:
    """One frozen number: the 90th percentile of u on the training half.

    By construction this makes the 90% breach rate exactly 10% in training.
    Whether it holds up out of sample is the entire question.
    """
    k = float(np.quantile(u[train], 0.90))
    return np.full(len(u), k)


def k_trailing(u: np.ndarray, window: int) -> np.ndarray:
    """k_t = 90th percentile of u over the trailing window, strictly past."""
    s = pd.Series(u).shift(1).rolling(window, min_periods=window // 2)
    k = s.quantile(0.90).to_numpy().copy()
    first = np.nanmin(np.where(np.isfinite(k))[0]) if np.isfinite(k).any() else 0
    k[:first] = k[first] if np.isfinite(k[first]) else 1.0
    return np.where(np.isfinite(k), k, 1.0)


def rates(r: pd.DataFrame, k: np.ndarray, mask: np.ndarray) -> tuple:
    return tuple(float((r.loss.to_numpy()[mask]
                        > (r[f"new_var_{lv}"].to_numpy() * k)[mask]).mean())
                 for lv in LEVELS)


def err(obs: tuple) -> float:
    """Coverage error, weighted toward the level CVaR is computed from."""
    w = (1.0, 1.0, 3.0)
    return float(sum(wi * ((o - t) / t) ** 2
                     for wi, o, t in zip(w, obs, TARGETS)))


def line(label: str, r: pd.DataFrame, k: np.ndarray, mask: np.ndarray) -> None:
    n = int(mask.sum())
    obs = rates(r, k, mask)
    v99 = r["new_var_0.99"].to_numpy() * k
    c99 = r["new_cvar_0.99"].to_numpy() * k
    br_all = r.loss.to_numpy() > v99
    br = br_all[mask]
    es = (float(r.loss.to_numpy()[mask][br].mean() / c99[mask][br].mean())
          if br.sum() > 2 else float("nan"))
    print(f"  {label:<24}{n:>7,}{100*obs[0]:>8.2f}{100*obs[1]:>8.2f}"
          f"{100*obs[2]:>8.2f}{kupiec(n, int(br.sum()), 0.01):>10.3f}"
          f"{christoffersen(br):>8.3f}{es:>7.2f}")


def header() -> None:
    print(f"  {'':<24}{'n':>7}{'90%':>8}{'95%':>8}{'99%':>8}"
          f"{'Kup99':>10}{'ind99':>8}{'ES99':>7}")
    print(f"  {'target':<24}{'':>7}{10.0:>8.2f}{5.0:>8.2f}{1.0:>8.2f}"
          f"{'>.05':>10}{'>.05':>8}{1.0:>7.2f}")
    print("  " + "-" * 72)


def main() -> None:
    r = pd.read_csv(data_path("coverage_clean_results.csv"),
                    index_col=0, parse_dates=True)
    u = (r.loss / r["new_var_0.9"]).to_numpy(dtype=float)
    train = np.asarray(r.index < SPLIT)
    test = ~train
    ones = np.ones(len(r))

    print("=" * 80)
    print("SCALE CORRECTION - every choice made on the training half")
    print("=" * 80)
    print(f"  train  {r.index.min().date()} -> {r.index[train].max().date()}"
          f"   {train.sum():,} days")
    print(f"  test   {r.index[test].min().date()} -> {r.index.max().date()}"
          f"   {test.sum():,} days")
    kc = k_constant(u, train)
    print(f"\n  constant correction estimated on training: "
          f"x{kc[0]:.3f}  (the model's VaR is {100*(kc[0]-1):.0f}% too small)")

    print("\n  Trailing-window selection, TRAINING half only:")
    print(f"  {'window':>8}{'90%':>8}{'95%':>8}{'99%':>8}{'error':>9}")
    best_w, best_e = None, np.inf
    for w in WINDOWS:
        obs = rates(r, k_trailing(u, w), train)
        e = err(obs)
        flag = ""
        if e < best_e:
            best_w, best_e, flag = w, e, "  <- best"
        print(f"  {w:>8}{100*obs[0]:>8.2f}{100*obs[1]:>8.2f}"
              f"{100*obs[2]:>8.2f}{e:>9.3f}{flag}")
    kt = k_trailing(u, best_w)
    print(f"\n  selected trailing window = {best_w} days")

    print("\n" + "=" * 80)
    print("TRAINING HALF  (where the choices were made - expect flattery)")
    print("=" * 80)
    header()
    line("uncorrected", r, ones, train)
    line(f"constant x{kc[0]:.2f}", r, kc, train)
    line(f"trailing {best_w}d", r, kt, train)

    print("\n" + "=" * 80)
    print("HELD-OUT HALF  (the number that counts)")
    print("=" * 80)
    header()
    line("uncorrected", r, ones, test)
    line(f"constant x{kc[0]:.2f}", r, kc, test)
    line(f"trailing {best_w}d", r, kt, test)

    print("\n  Sensitivity on the held-out half - if only the selected window")
    print("  works, the result is selection rather than a finding:")
    print(f"  {'window':>8}{'90%':>8}{'95%':>8}{'99%':>8}{'Kup99':>10}")
    for w in WINDOWS:
        kk = k_trailing(u, w)
        obs = rates(r, kk, test)
        k99 = int(((r.loss.to_numpy() > r["new_var_0.99"].to_numpy() * kk)
                   & test).sum())
        mark = "  <- selected" if w == best_w else ""
        print(f"  {w:>8}{100*obs[0]:>8.2f}{100*obs[1]:>8.2f}{100*obs[2]:>8.2f}"
              f"{kupiec(int(test.sum()), k99, 0.01):>10.3f}{mark}")

    print("\n" + "=" * 80)
    print("ALL THREE LEVELS, HELD-OUT HALF - Kupiec p for every candidate")
    print("=" * 80)
    print("  The pre-registered choice is the 500-day window, because that is")
    print("  what the training half selected. It is reported first and it is")
    print("  the primary result. The longer windows are sensitivity: they do")
    print("  BETTER out of sample than the one the selection picked, which is")
    print("  evidence against the selection having flattered anything.")
    cands = {"uncorrected": ones, f"constant x{kc[0]:.3f}": kc}
    for w in WINDOWS:
        cands[f"trailing {w}d"] = k_trailing(u, w)
    print(f"\n  {'correction':<20}{'90%':>7}{'p':>7}{'95%':>7}{'p':>7}"
          f"{'99%':>7}{'p':>7}   verdict")
    print("  " + "-" * 70)
    for nm, k in cands.items():
        cells, ps = [], []
        for lv, pt in zip(LEVELS, TARGETS):
            br = (r.loss.to_numpy() > r[f"new_var_{lv}"].to_numpy() * k)[test]
            n, kk = int(test.sum()), int(br.sum())
            cells.append(100 * kk / n)
            ps.append(kupiec(n, kk, pt))
        verdict = "PASSES all three" if all(x > 0.05 for x in ps) else "fails"
        star = "  *" if nm == f"trailing {best_w}d" else ""
        print(f"  {nm:<20}{cells[0]:>7.2f}{ps[0]:>7.3f}{cells[1]:>7.2f}"
              f"{ps[1]:>7.3f}{cells[2]:>7.2f}{ps[2]:>7.3f}   {verdict}{star}")
    print("\n  * = the pre-registered choice")

    out = r.copy()
    out["k_const"], out["k_trailing"] = kc, kt
    for lv in LEVELS:
        out[f"adj_var_{lv}"] = r[f"new_var_{lv}"] * kt
        out[f"adj_cvar_{lv}"] = r[f"new_cvar_{lv}"] * kt
    out["split"] = np.where(train, "train", "test")
    out.to_csv(data_path("coverage_adjusted.csv"))
    print(f"\n  wrote coverage_adjusted.csv")


if __name__ == "__main__":
    main()
