"""
Step 4 - The experiment grid.

For every (process, estimation window, confidence level) cell, roll each
estimator forward through simulated data whose true VaR is known exactly, and
record how far the estimate sits from truth and how often losses breach it.

WHAT IS MEASURED, AND WHY EACH ONE
----------------------------------
bias        mean (estimate / truth - 1). Systematic over- or under-statement.
rmse        root mean square of the same relative error. A method can be
            unbiased and still useless if it is noisy; this separates the two.
breach      share of days where the realised loss exceeded the estimate. The
            coverage test, but now against KNOWN truth and with enough
            observations to have power.
indep_p     Christoffersen independence. Do breaches cluster? A model with the
            right average rate can still be wrong in episodes.
fallback    share of GARCH fits that failed and fell back to EWMA. Reported so
            a convergence problem cannot masquerade as a modelling result.

INFERENCE
---------
Consecutive test points inside one replication share nearly all of their
estimation window, so they are far from independent. Metrics are therefore
computed PER REPLICATION and averaged across replications, with the standard
error taken from the across-replication spread. Replications are independent by
construction, so that standard error is honest; pooling every test point and
treating them as independent would understate it by roughly sqrt(T).

COST
----
The binding cost is GARCH fitting. Two savings, both exact rather than
approximations: all three confidence levels reuse one volatility estimate
(the quantile changes, the fit does not), and GARCH is refitted every
`refit_every` test points with the recursion carried forward in between -
which is also how the model would be operated in practice.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

warnings.filterwarnings("ignore")

from dgp import Process, default_grid                      # noqa: E402
from truth import conditional_truth, innovation_var_cvar   # noqa: E402
from var_estimators import (default_estimators, ewma_sigma_last,   # noqa: E402
                            fit_garch_forecast, normal_tail, student_tail)

LEVELS = (0.90, 0.95, 0.99)


# ---------------------------------------------------------------------------
# Design: what can this grid actually detect?
# ---------------------------------------------------------------------------

def power_report(n_test: int, levels=LEVELS) -> pd.DataFrame:
    """Resolution of the breach-rate estimate at a given number of test points.

    Run BEFORE the grid. If the design cannot separate a correct model from one
    that is 50% wrong, the grid is not worth running - which is exactly the
    conclusion the monthly coverage test reached on real data.
    """
    rows = []
    for lv in levels:
        p = 1 - lv
        se = math.sqrt(p * (1 - p) / n_test)
        rows.append({
            "level": lv, "expected_rate": p, "n_test": n_test,
            "se_of_rate": se,
            "detectable_at_80pct_power": 2.8 * se,
            "as_pct_of_target": 2.8 * se / p,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One replication
# ---------------------------------------------------------------------------

def _tail_fn(kind: str, q: float, nu: Optional[float]):
    if kind == "t" and nu is not None and np.isfinite(nu):
        return student_tail(q, nu)
    return normal_tail(q)


def run_replication(proc: Process, window: int, n_test: int,
                    rng: np.random.Generator, refit_every: int = 20,
                    k_sigma: float = 4.0) -> Dict[str, Dict[str, float]]:
    """Roll every estimator across n_test points of one simulated path.

    Returns per-estimator, per-level summary statistics for this replication.
    """
    n = window + n_test
    if proc.is_iid:
        x = proc.simulate(n, rng)
        sigma_true = np.ones(n)
    else:
        x, sigma_true = proc.simulate(n, rng, return_sigma=True)

    # exact conditional truth at every test point
    truth_mult = {q: innovation_var_cvar(proc, q) for q in LEVELS}

    est_names = ["naive", "k_sigma", "sma_normal", "ewma_normal",
                 "garch_normal", "garch_t"]
    acc = {e: {q: {"ev": [], "ec": [], "breach": []} for q in LEVELS}
           for e in est_names}
    n_fallback = 0

    garch_state: Dict[str, Optional[Dict[str, float]]] = {"normal": None, "t": None}

    for i in range(n_test):
        lo, hi = i, i + window
        w = x[lo:hi]
        realized_loss = -x[hi]
        s_true = sigma_true[hi]

        sd = float(np.std(w))
        ewma_s = ewma_sigma_last(w, 0.94)
        loss_sorted = np.sort(-w)

        # GARCH: refit on a cadence, carry the recursion forward between fits
        for dist in ("normal", "t"):
            if i % refit_every == 0 or garch_state[dist] is None:
                g = fit_garch_forecast(w, dist, fast=True)
                if not g.get("ok"):
                    n_fallback += 1
                    g = {"ok": 1.0, "sigma": ewma_s, "nu": float("nan"),
                         "fallback": True}
                garch_state[dist] = g
            else:
                prev = garch_state[dist]
                a, b = prev.get("alpha", 0.08), prev.get("beta", 0.90)
                if np.isfinite(a) and np.isfinite(b) and a + b < 1:
                    uncond = float(np.var(w))
                    om = max(uncond * (1 - a - b), 1e-300)
                    v = om + a * x[hi - 1] ** 2 + b * prev["sigma"] ** 2
                    garch_state[dist] = {**prev, "sigma": math.sqrt(max(v, 1e-300))}

        for q in LEVELS:
            tv, tc = truth_mult[q]
            true_var, true_cvar = s_true * tv, s_true * tc

            # --- naive historical
            j = min(int(math.ceil(q * loss_sorted.size)) - 1, loss_sorted.size - 1)
            v = float(loss_sorted[j]); c = float(loss_sorted[j:].mean())
            _rec(acc["naive"][q], v, c, true_var, true_cvar, realized_loss)

            # --- k-sigma (VaR only; it is a sizing rule, not a distribution)
            _rec(acc["k_sigma"][q], k_sigma * sd, float("nan"),
                 true_var, true_cvar, realized_loss)

            # --- moving average + normal
            zv, zc = normal_tail(q)
            _rec(acc["sma_normal"][q], zv * sd, zc * sd,
                 true_var, true_cvar, realized_loss)

            # --- EWMA + normal
            _rec(acc["ewma_normal"][q], zv * ewma_s, zc * ewma_s,
                 true_var, true_cvar, realized_loss)

            # --- GARCH + normal
            gn = garch_state["normal"]
            _rec(acc["garch_normal"][q], zv * gn["sigma"], zc * gn["sigma"],
                 true_var, true_cvar, realized_loss)

            # --- GARCH + Student-t
            gt = garch_state["t"]
            tzv, tzc = _tail_fn("t", q, gt.get("nu"))
            _rec(acc["garch_t"][q], tzv * gt["sigma"], tzc * gt["sigma"],
                 true_var, true_cvar, realized_loss)

    out = {}
    for e in est_names:
        out[e] = {}
        for q in LEVELS:
            d = acc[e][q]
            ev = np.array(d["ev"], dtype=float)
            ec = np.array(d["ec"], dtype=float)
            br = np.array(d["breach"], dtype=float)
            out[e][q] = {
                "bias": float(np.nanmean(ev)),
                "rmse": float(np.sqrt(np.nanmean(ev ** 2))),
                "cvar_bias": float(np.nanmean(ec)) if np.isfinite(ec).any() else float("nan"),
                "breach": float(np.mean(br)),
                "indep_lr": _christoffersen(br),
                "fallback": n_fallback / max(1, 2 * (n_test // refit_every + 1)),
            }
    return out


def _rec(slot, var_est, cvar_est, true_var, true_cvar, realized_loss):
    slot["ev"].append(var_est / true_var - 1.0)
    slot["ec"].append(cvar_est / true_cvar - 1.0 if np.isfinite(cvar_est) else np.nan)
    slot["breach"].append(1.0 if realized_loss > var_est else 0.0)


def _christoffersen(b: np.ndarray) -> float:
    b = b.astype(int)
    n00 = n01 = n10 = n11 = 0
    for a, c in zip(b[:-1], b[1:]):
        if a == 0 and c == 0: n00 += 1
        elif a == 0 and c == 1: n01 += 1
        elif a == 1 and c == 0: n10 += 1
        else: n11 += 1
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return float("nan")
    p01 = n01 / (n00 + n01); p11 = n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    if min(p01, p11, p) <= 0 or max(p01, p11, p) >= 1:
        return float("nan")
    ll0 = (n00 + n10) * math.log(1 - p) + (n01 + n11) * math.log(p)
    ll1 = (n00 * math.log(1 - p01) + n01 * math.log(p01)
           + n10 * math.log(1 - p11) + n11 * math.log(p11))
    return 2 * (ll1 - ll0)


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

def run_grid(processes: List[Process], windows: List[int], n_reps: int,
             n_test: int, seed: int = 20261013, refit_every: int = 20,
             verbose: bool = True) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    total = len(processes) * len(windows)
    done = 0
    for proc in processes:
        for W in windows:
            reps = []
            for r in range(n_reps):
                rng = np.random.default_rng(seed + 1000 * r + W)
                reps.append(run_replication(proc, W, n_test, rng, refit_every))
            for est in reps[0].keys():
                for q in LEVELS:
                    vals = {k: np.array([rep[est][q][k] for rep in reps], dtype=float)
                            for k in reps[0][est][q].keys()}
                    rows.append({
                        "process": proc.name, "family": proc.family,
                        "tail_index": proc.tail_index,
                        "persistence": (proc.params.get("alpha", 0) + proc.params.get("beta", 0)
                                        if proc.family == "garch" else float("nan")),
                        "lam": proc.params.get("lam", float("nan")),
                        "exkurt": proc.moments()["exkurt"],
                        "window": W, "level": q, "estimator": est,
                        "n_reps": n_reps, "n_test": n_test,
                        "bias": float(np.nanmean(vals["bias"])),
                        "bias_se": float(np.nanstd(vals["bias"], ddof=1) / math.sqrt(n_reps)),
                        "rmse": float(np.nanmean(vals["rmse"])),
                        "cvar_bias": float(np.nanmean(vals["cvar_bias"])),
                        "breach": float(np.nanmean(vals["breach"])),
                        "breach_se": float(np.nanstd(vals["breach"], ddof=1) / math.sqrt(n_reps)),
                        "indep_lr": float(np.nanmean(vals["indep_lr"])),
                        "fallback": float(np.nanmean(vals["fallback"])),
                    })
            done += 1
            if verbose:
                el = time.time() - t0
                print(f"  [{done}/{total}] {proc.name:<28} W={W:<5} "
                      f"{el:6.0f}s elapsed, ~{el/done*(total-done):5.0f}s left",
                      flush=True)
    return pd.DataFrame(rows)


def select_processes(which: str) -> List[Process]:
    g = default_grid()
    if which == "all":
        return [p for grp in g.values() for p in grp]
    if which == "smoke":
        return [g["control"][0], g["kurtosis"][4], g["skewness"][0], g["clustering"][2]]
    return g[which]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 4 - VaR estimator grid")
    ap.add_argument("--which", default="smoke",
                    help="all | smoke | control | kurtosis | skewness | clustering | clustering_fat")
    ap.add_argument("--windows", default="125,250,500,1000,2000")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--refit-every", type=int, default=20)
    ap.add_argument("--out", default="grid_results.csv")
    a = ap.parse_args()

    windows = [int(w) for w in a.windows.split(",")]
    procs = select_processes(a.which)

    print("DESIGN RESOLUTION (per cell, before running)")
    print(power_report(a.reps * a.test).round(5).to_string(index=False))
    print(f"\n{len(procs)} processes x {len(windows)} windows x {a.reps} reps "
          f"x {a.test} test points")
    print(f"= {len(procs)*len(windows)*a.reps*a.test:,} estimator evaluations per method\n")

    df = run_grid(procs, windows, a.reps, a.test, refit_every=a.refit_every)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(df)} rows)")
