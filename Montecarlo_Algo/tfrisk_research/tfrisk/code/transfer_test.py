"""
Step 6 - The transfer test.

The synthetic study is only worth anything if its predictions hold on the real
thing. So: take the parameters measured on the actual strategy, find the
matching cell in the synthetic grid, read off what the study SAYS the coverage
error should be, and compare against what the real coverage test ACTUALLY
produced. Agreement validates the study's other recommendations. Disagreement
is itself the finding - it means the synthetic processes are missing something.

Measured on the real data (Phase 1 + step 6):
    tail index of filtered residuals    ~6.9 portfolio, 5.9 median per-market
    volatility persistence (a+b)         0.9836 median
    skewness of filtered residuals      -0.213 portfolio, -0.214 median market

Real coverage test, matched sample of 1,972 observations:
    OLD model (student-t + EWMA)   9.3 / 4.7 / 1.5   vs targets 10 / 5 / 1
    NEW model (FHS)                7.0 / 3.5 / 1.4
    shape ratio VaR99/VaR90        modelled 2.09, realised 3.05

    python transfer_test.py
"""

from __future__ import annotations

import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

warnings.filterwarnings("ignore")

from dgp import Process                                      # noqa: E402
from truth import innovation_var_cvar                         # noqa: E402
from paths import data_path, fig_path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

NAMES = {"naive": "naive hist", "k_sigma": "4-sigma", "sma_normal": "moving avg",
         "ewma_normal": "EWMA", "garch_normal": "GARCH-N", "garch_t": "GARCH-t"}
ORDER = ["naive hist", "moving avg", "EWMA", "GARCH-N", "GARCH-t"]

# measured on the real strategy
REAL = {"tail_index": 6.9, "persistence": 0.9836, "skew": -0.213}
REAL_COVERAGE = {"OLD (t + EWMA)": (9.3, 4.7, 1.5), "NEW (FHS)": (7.0, 3.5, 1.4)}
TARGETS = (10.0, 5.0, 1.0)
REAL_SHAPE = {"modelled": 2.09, "realised": 3.05}


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. which synthetic cell actually matches the real data
# ---------------------------------------------------------------------------

def cell_params(process_name: str) -> dict:
    """Recover (nu, lam) from the process label, e.g. skt7/-0.08."""
    tag = process_name.split("skt")[1].rstrip(")")
    nu, lam = tag.split("/")
    return {"nu": float(nu), "lam": float(lam)}


def match_cell(d: pd.DataFrame, rng) -> str:
    """Pick the cell closest to the real data on BOTH tail index and skewness.

    Hansen's lam is a shape parameter, not a skewness, so cells are ranked by
    their actual simulated skewness rather than by lam directly. Matching on
    skew alone would pick the nu=10 cell, whose tail is visibly thinner than
    the measured 6.9 - the two parameters have to be traded off jointly, and
    the distance below is scaled so one unit of each is comparable (0.1 of
    skew against 1.0 of tail index, roughly their measurement uncertainty).
    """
    from dgp import hansen_rvs

    rows = []
    for nm in d.process.unique():
        p = cell_params(nm)
        z = hansen_rvs(400_000, p["nu"], p["lam"], rng)
        sk = float(sps.skew(z))
        dist = math.hypot((sk - REAL["skew"]) / 0.10,
                          (p["nu"] - REAL["tail_index"]) / 1.0)
        rows.append((nm, p["nu"], p["lam"], sk, dist))

    print(f"  {'cell':<36}{'tail idx':>9}{'lam':>7}{'innov skew':>12}{'distance':>10}")
    for nm, nu, lam, sk, dist in rows:
        print(f"  {nm:<36}{nu:>9.1f}{lam:>7.2f}{sk:>12.3f}{dist:>10.2f}")
    best = min(rows, key=lambda r: r[4])[0]
    print(f"\n  measured on real filtered residuals: skew {REAL['skew']:+.3f}, "
          f"tail index {REAL['tail_index']:.1f}, persistence {REAL['persistence']:.4f}")
    print(f"  grid persistence 0.984 vs real 0.9836 - matched by construction")
    print(f"  closest synthetic cell -> {best}")
    return best


# ---------------------------------------------------------------------------
# 2. shape ratio: the diagnostic that separates the failure modes
# ---------------------------------------------------------------------------

def shape_ratios(d: pd.DataFrame, cell: str) -> pd.DataFrame:
    """VaR99 / VaR90 for truth and for each estimator, by window.

    A single breach rate can be wrong for two opposite reasons - the whole
    curve too low, or the curve the right height but the wrong SHAPE. The ratio
    of the 99% number to the 90% number separates them: it is scale-free, so it
    isolates tail shape from the level of volatility.
    """
    p = cell_params(cell)
    proc = Process("garch", {"alpha": 0.08, "beta": 0.904, "innov": "skew_t", **p})
    v90 = innovation_var_cvar(proc, 0.90)[0]
    v99 = innovation_var_cvar(proc, 0.99)[0]
    true_ratio = v99 / v90

    k = d[d.process == cell]
    b90 = k[k.level == 0.90].pivot_table(index="estimator", columns="window", values="bias")
    b99 = k[k.level == 0.99].pivot_table(index="estimator", columns="window", values="bias")
    # estimator VaR = truth * (1 + relative bias), so the ratio follows directly
    ratio = (v99 * (1 + b99)) / (v90 * (1 + b90))
    ratio.index = [NAMES.get(i, i) for i in ratio.index]
    ratio = ratio.reindex([e for e in ORDER if e in ratio.index])
    return true_ratio, ratio


# ---------------------------------------------------------------------------
# 3. the comparison the whole step exists for
# ---------------------------------------------------------------------------

def predicted_vs_actual(d: pd.DataFrame, cell: str):
    k = d[d.process == cell]
    piv = k.pivot_table(index=["estimator", "window"], columns="level", values="breach")
    piv = piv * 100
    piv.columns = ["b90", "b95", "b99"]
    piv = piv.reset_index()
    piv["estimator"] = piv.estimator.map(lambda s: NAMES.get(s, s))
    piv = piv[piv.estimator != "4-sigma"]

    print(f"  {'method':<13}{'window':>8}{'90%':>8}{'95%':>8}{'99%':>8}"
          f"{'b99/b90':>9}   verdict")
    print(f"  {'target':<13}{'':>8}{TARGETS[0]:>8.1f}{TARGETS[1]:>8.1f}"
          f"{TARGETS[2]:>8.1f}{0.10:>9.2f}")
    print("  " + "-" * 76)
    for est in ORDER:
        for _, r in piv[piv.estimator == est].iterrows():
            # A scale failure moves all three levels the same way. A shape
            # failure moves the two ends in OPPOSITE directions, so the ratio
            # of the 99% breach rate to the 90% one is the diagnostic: target
            # 0.10, and anything materially above it means the estimate is too
            # flat - comfortable at 90%, breaking through at 99%.
            lo, hi = r.b90 - TARGETS[0], r.b99 - TARGETS[2]
            br = r.b99 / r.b90
            if abs(lo) <= 1.2 and abs(hi) <= 0.3:
                verdict = "calibrated"
            elif br > 0.16 and lo < 0:
                verdict = "TOO FLAT: short at 90, breaks at 99"
            elif br > 0.16:
                verdict = "too flat and too thin overall"
            elif lo < -1.2 and hi < 0:
                verdict = "conservative at both ends"
            elif lo > 1.2:
                verdict = "too thin overall"
            else:
                verdict = "near target"
            print(f"  {est:<13}{int(r['window']):>8}{r.b90:>8.2f}{r.b95:>8.2f}"
                  f"{r.b99:>8.2f}{br:>9.2f}   {verdict}")


def conditional_vs_unconditional(cell: str, rng):
    """The comparison Phase 1 got wrong, and why it matters.

    Phase 1 compared the model's OWN VaR99/VaR90 (2.09) against the ratio of
    the realised loss quantiles pooled over the whole sample, and read the gap
    as the model being too flat in the tail. Those are different objects. The
    model's ratio is conditional - one day's tail shape given that day's
    volatility. Pooling realised losses across days mixes high- and low-
    volatility days together, which fattens the tail regardless of whether the
    model is right.

    The synthetic process settles how big that effect should be, because here
    both objects are known exactly. If the pooled/conditional spread on
    synthetic data matches the spread seen on real data, the Phase 1 "gap" was
    an artefact of the comparison, not a defect in the model.
    """
    p = cell_params(cell)
    proc = Process("garch", {"alpha": 0.08, "beta": 0.904, "innov": "skew_t", **p})
    cond = (innovation_var_cvar(proc, 0.99)[0] / innovation_var_cvar(proc, 0.90)[0])
    x, s = proc.simulate(4_000_000, rng, return_sigma=True)
    L = -x
    std = float(np.quantile(L / s, 0.99) / np.quantile(L / s, 0.90))
    unc = float(np.quantile(L, 0.99) / np.quantile(L, 0.90))

    print("  SYNTHETIC matched cell, where both objects are known exactly:")
    print(f"    conditional (analytic innovation ratio)      {cond:.2f}")
    print(f"    standardised by the true sigma, 4M draws     {std:.2f}")
    print(f"    pooled across all days (unconditional)       {unc:.2f}")

    real = real_shape_ratios()
    print("\n  REAL data, 1,972 matched observations, FHS model:")
    print(f"    losses standardised by the model's own VaR90 {real['std']:.2f}")
    print(f"    pooled across all days                       {real['pooled']:.2f}"
          f"   [block bootstrap {real['ci'][0]:.2f}, {real['ci'][1]:.2f}]")
    print(f"    the model's own ratio (what Phase 1 quoted)  {real['model']:.2f}")

    print(f"\n  pooled / conditional spread:  synthetic {unc / cond:.2f}x,"
          f"  real {real['pooled'] / real['model']:.2f}x")


def real_shape_ratios() -> dict:
    """Recompute the real shape ratios from the saved coverage output.

    Phase 1's write-up records a realised ratio of 3.05. That number is not
    reproducible from the saved artifacts by any definition tried here (pooled,
    per-run, standardised, mean or median modelled); the pooled figure is 2.54
    and the per-run figure 2.49. The recomputed value is used below and the
    discrepancy is flagged rather than quietly reconciled.
    """
    path = data_path("coverage_fhs_results.csv")
    d = pd.read_csv(path)
    L = -d.realized_loss.to_numpy()
    pooled = float(np.quantile(L, 0.99) / np.quantile(L, 0.90))
    z = L / d["var_0.9"].to_numpy()
    std = float(np.quantile(z, 0.99) / np.quantile(z, 0.90))
    model = float(d["var_0.99"].mean() / d["var_0.9"].mean())

    # block bootstrap: 40-day blocks, so clustering is preserved and the
    # interval is not falsely narrow
    rng = np.random.default_rng(7)
    n, B = len(L), 40
    draws = []
    for _ in range(4000):
        st = rng.integers(0, n - B, n // B)
        s = np.concatenate([L[i:i + B] for i in st])
        draws.append(np.quantile(s, 0.99) / np.quantile(s, 0.90))
    draws = np.asarray(draws)
    return {"pooled": pooled, "std": std, "model": model, "n": n,
            "ci": (float(np.percentile(draws, 2.5)),
                   float(np.percentile(draws, 97.5)))}


def nearest_real(d: pd.DataFrame, cell: str):
    """Which synthetic cell/window reproduces the REAL coverage numbers."""
    k = d[d.process == cell]
    piv = k.pivot_table(index=["estimator", "window"], columns="level", values="breach") * 100
    piv.columns = ["b90", "b95", "b99"]
    piv = piv.reset_index()
    piv["estimator"] = piv.estimator.map(lambda s: NAMES.get(s, s))

    for label, (r90, r95, r99) in REAL_COVERAGE.items():
        piv["dist"] = np.sqrt((piv.b90 - r90) ** 2 + (piv.b95 - r95) ** 2
                              + 4 * (piv.b99 - r99) ** 2)
        best = piv.sort_values("dist").iloc[0]
        print(f"\n  real {label}: {r90:.1f} / {r95:.1f} / {r99:.1f}")
        print(f"    closest synthetic: {best.estimator} at window "
              f"{int(best['window'])} -> {best.b90:.2f} / {best.b95:.2f} / "
              f"{best.b99:.2f}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(606)
    d = pd.read_csv(data_path("grid_realistic.csv"))
    print(f"loaded {len(d)} rows from grid_realistic.csv "
          f"({d.process.nunique()} cells x {d.window.nunique()} windows "
          f"x {d.level.nunique()} levels x {d.estimator.nunique()} estimators)")

    banner("1. MATCHING THE SYNTHETIC GRID TO THE REAL DATA")
    cell = match_cell(d, rng)

    banner("2. TAIL SHAPE  (VaR99 / VaR90)")
    true_ratio, ratio = shape_ratios(d, cell)
    print(f"  TRUE shape ratio for this process: {true_ratio:.2f}")
    print(f"  Phase 1 on real data: modelled {REAL_SHAPE['modelled']:.2f}, "
          f"realised {REAL_SHAPE['realised']:.2f}\n")
    print(f"  {'method':<13}" + "".join(f"{int(w):>8}" for w in ratio.columns))
    for est, row in ratio.iterrows():
        print(f"  {est:<13}" + "".join(f"{v:>8.2f}" for v in row.to_numpy(dtype=float)))
    print(f"\n  A normal quantile forces the ratio to norm.ppf(.99)/norm.ppf(.90)")
    print(f"  = {sps.norm.ppf(0.99) / sps.norm.ppf(0.90):.2f} regardless of the data. "
          f"The truth here is {true_ratio:.2f}.")

    banner("3. PREDICTED COVERAGE AT THE MATCHED CELL")
    predicted_vs_actual(d, cell)

    banner("4. CONDITIONAL vs UNCONDITIONAL - the Phase 1 comparison error")
    conditional_vs_unconditional(cell, rng)

    banner("5. WHICH SYNTHETIC SETUP REPRODUCES THE REAL COVERAGE")
    nearest_real(d, cell)
