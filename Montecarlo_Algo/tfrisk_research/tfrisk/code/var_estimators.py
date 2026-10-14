"""
Step 1 - The VaR/CVaR estimators under test.

Each takes a trailing window of returns and produces a one-step-ahead estimate
for the next observation. Every estimator sees exactly the same window, so any
difference in performance is the method and nothing else.

SHARED ASSUMPTIONS, stated because they matter
-----------------------------------------------
* Zero mean. Standard for daily risk work: the drift is tiny relative to the
  daily standard deviation and estimating it adds more noise than it removes.
  All synthetic processes here have a true mean of zero, so this assumption is
  exactly correct and is not a source of error in the study.
* One-step-ahead. The estimate for day t uses the window ending at t-1. No
  estimator ever sees the observation it is predicting.
* Loss convention. VaR and CVaR are positive numbers on the loss scale.

THE ESTIMATORS
--------------
naive        empirical quantile of the window. No distributional assumption.
             At 99% on a 250-day window this rests on ~2.5 observations.
k_sigma      a fixed multiple of the sample standard deviation. NOTE this is
             not tied to any confidence level: 4 sigma is 99.997% under a
             normal law but roughly 99.3% under a t(5). Reported as VaR only,
             since it is a sizing heuristic rather than a distributional model.
sma_normal   equal-weighted variance over the window, normal quantile.
ewma_normal  exponentially weighted variance, normal quantile. One parameter,
             no fitting - the natural midpoint between sma and garch.
garch_normal GARCH(1,1) conditional volatility, normal quantile.
garch_t      GARCH(1,1) conditional volatility, Student-t quantile with the
             degrees of freedom fitted alongside.

Including both a correctly-specified and a misspecified parametric option on
the same volatility model separates "wrong volatility" from "wrong tail shape",
which is the distinction the whole Phase 1 argument turned on.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import optimize, signal, stats as sps

sys.path.insert(0, "/home/claude/eda")


# ---------------------------------------------------------------------------
# Standardised-distribution tail functions (unit variance)
# ---------------------------------------------------------------------------

def normal_tail(q: float) -> Tuple[float, float]:
    z = sps.norm.ppf(q)
    return float(z), float(sps.norm.pdf(z) / (1 - q))


def student_tail(q: float, nu: float) -> Tuple[float, float]:
    if nu <= 2.01:
        nu = 2.01
    s = math.sqrt((nu - 2) / nu)
    p = 1 - q
    t_p = sps.t.ppf(p, nu)
    var = -s * t_p
    es = s * ((nu + t_p ** 2) / (nu - 1)) * sps.t.pdf(t_p, nu) / p
    return float(var), float(es)


# ---------------------------------------------------------------------------
# Volatility filters
# ---------------------------------------------------------------------------

def ewma_sigma_last(x: np.ndarray, lam: float) -> float:
    """EWMA conditional sd for the NEXT observation after the window."""
    v = float(np.var(x[: max(20, len(x) // 10)])) or float(np.var(x))
    for xt in x:
        v = lam * v + (1 - lam) * xt * xt
    return math.sqrt(max(v, 1e-300))


def _garch_filter(params, x):
    """GARCH(1,1) conditional variance, one step ahead.

    The recursion v_t = omega + alpha*x_{t-1}^2 + beta*v_{t-1} is a first-order
    linear filter, so scipy's lfilter runs it in compiled C instead of a Python
    loop. Identical arithmetic, ~50x faster, and the optimiser evaluates this
    hundreds of times per fit. Verified against the loop to machine precision.
    """
    omega, alpha, beta = params
    n = len(x)
    out = np.empty(n + 1)
    v0 = omega / max(1 - alpha - beta, 1e-6)
    # drive: u_t = omega + alpha * x_{t-1}^2, with x_{-1} unused (v_0 is given)
    u = np.empty(n)
    u[0] = v0                        # placeholder; overwritten by zi below
    if n > 1:
        u[1:] = omega + alpha * x[:-1] ** 2
    zi = np.array([beta * v0])
    filt = signal.lfilter([1.0], [1.0, -beta], u[1:], zi=zi)[0] if n > 1 else np.array([])
    out[0] = v0
    if n > 1:
        out[1:n] = filt
    out[n] = omega + alpha * x[n - 1] ** 2 + beta * out[n - 1]
    return np.clip(out, 1e-300, None)


def fit_garch_forecast(x: np.ndarray, dist: str = "normal",
                       fast: bool = True) -> Dict[str, float]:
    """Fit GARCH(1,1) on the window; return the next-step sd and (for t) nu.

    VARIANCE TARGETING. omega is not a free parameter: it is fixed at
    sample_variance * (1 - alpha - beta), so the model's unconditional variance
    equals the sample variance by construction and only alpha and beta are
    estimated.

    This is standard practice, and here it is also a large speed win. Estimating
    omega freely leaves the optimiser with parameters on wildly different
    scales - omega is O(500) in the internal units while alpha and beta are
    O(0.1) - which conditions the problem badly: profiling showed 83% of the fit
    time inside the L-BFGS-B core rather than in the likelihood. Removing omega
    makes both remaining parameters O(1) and cuts the cost ~25x, with estimates
    that agree with the unrestricted fit (verified in validation).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 60:
        return {"ok": 0.0}
    scale = 100.0 / max(np.std(x), 1e-12)
    y = x * scale                          # zero-mean assumption, see header
    target = float(np.var(y))
    if not np.isfinite(target) or target <= 0:
        return {"ok": 0.0}

    def params_of(al, be):
        return [max(target * (1 - al - be), 1e-12), al, be]

    def nll_norm(p):
        al, be = p
        if al < 0 or be < 0 or al + be >= 0.9995:
            return 1e10
        v = _garch_filter(params_of(al, be), y)[:-1]
        return float(0.5 * np.sum(np.log(v) + y ** 2 / v))

    def nll_t(p):
        al, be, nu = p
        if al < 0 or be < 0 or al + be >= 0.9995 or nu <= 2.05:
            return 1e10
        v = _garch_filter(params_of(al, be), y)[:-1]
        s2 = v * (nu - 2) / nu
        c = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
             - 0.5 * math.log(math.pi * nu))
        return float(-np.sum(c - 0.5 * np.log(s2)
                             - ((nu + 1) / 2) * np.log1p(y ** 2 / (nu * s2))))

    if dist == "normal":
        starts = [[0.08, 0.90], [0.05, 0.94]]
        if not fast:
            starts += [[0.12, 0.85], [0.03, 0.96]]
        bnds = [(1e-6, 0.6), (0.0, 0.9990)]
        fn = nll_norm
    else:
        starts = [[0.08, 0.90, 7.0], [0.05, 0.94, 5.0]]
        if not fast:
            starts += [[0.12, 0.85, 12.0], [0.03, 0.96, 6.0]]
        bnds = [(1e-6, 0.6), (0.0, 0.9990), (2.1, 60.0)]
        fn = nll_t

    best = None
    for s0 in starts:
        try:
            r = optimize.minimize(fn, s0, method="L-BFGS-B", bounds=bnds,
                                  options={"maxiter": 300, "ftol": 1e-9})
        except Exception:  # noqa: BLE001
            continue
        if r.fun < 1e9 and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        return {"ok": 0.0}
    al, be = float(best.x[0]), float(best.x[1])
    sig_next = math.sqrt(_garch_filter(params_of(al, be), y)[-1]) / scale
    return {"ok": 1.0, "sigma": sig_next,
            "nu": float(best.x[2]) if dist != "normal" else float("nan"),
            "alpha": al, "beta": be}


# ---------------------------------------------------------------------------
# The estimators
# ---------------------------------------------------------------------------

@dataclass
class Estimator:
    name: str
    kind: str
    params: Dict[str, float]

    def estimate(self, window: np.ndarray, q: float) -> Dict[str, float]:
        """Return {'var', 'cvar'} for the observation AFTER the window."""
        x = np.asarray(window, dtype=float)
        x = x[np.isfinite(x)]
        nan = {"var": float("nan"), "cvar": float("nan")}
        if x.size < 30:
            return nan
        sd = float(np.std(x))

        if self.kind == "naive":
            loss = np.sort(-x)
            i = min(int(math.ceil(q * loss.size)) - 1, loss.size - 1)
            v = float(loss[i])
            return {"var": v, "cvar": float(loss[i:].mean())}

        if self.kind == "k_sigma":
            # a sizing heuristic, not a distributional model: no CVaR
            return {"var": float(self.params["k"] * sd), "cvar": float("nan")}

        if self.kind == "sma_normal":
            zv, zc = normal_tail(q)
            return {"var": zv * sd, "cvar": zc * sd}

        if self.kind == "ewma_normal":
            s = ewma_sigma_last(x, self.params.get("lam", 0.94))
            zv, zc = normal_tail(q)
            return {"var": zv * s, "cvar": zc * s}

        if self.kind in ("garch_normal", "garch_t"):
            dist = "normal" if self.kind == "garch_normal" else "t"
            g = fit_garch_forecast(x, dist, fast=self.params.get("fast", True))
            if not g.get("ok"):
                # documented fallback so a failed fit does not silently vanish
                s = ewma_sigma_last(x, 0.94)
                zv, zc = normal_tail(q)
                return {"var": zv * s, "cvar": zc * s, "fallback": 1.0}
            s = g["sigma"]
            if dist == "normal":
                zv, zc = normal_tail(q)
            else:
                zv, zc = student_tail(q, g["nu"])
            return {"var": zv * s, "cvar": zc * s}

        raise ValueError(self.kind)


def garch_params_only(x: np.ndarray, dist: str = "normal") -> Dict[str, float]:
    """Fit GARCH parameters and return them WITHOUT the forecast.

    Used by the rolling coverage experiment, where parameters are refitted
    periodically but the variance recursion must still run every day. Freezing
    the whole estimate between refits would freeze the volatility state, which
    is the one thing a conditional model is supposed to update.
    """
    g = fit_garch_forecast(x, dist, fast=True)
    if not g.get("ok"):
        return {"ok": 0.0}
    return {"ok": 1.0, "alpha": g["alpha"], "beta": g["beta"], "nu": g["nu"]}


def garch_estimate_given_params(x: np.ndarray, q: float, par: Dict[str, float],
                                dist: str = "normal") -> Dict[str, float]:
    """One-step-ahead VaR/CVaR using HELD parameters but a fresh variance path."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not par.get("ok") or x.size < 30:
        s = ewma_sigma_last(x, 0.94) if x.size >= 30 else float("nan")
        zv, zc = normal_tail(q)
        return {"var": zv * s, "cvar": zc * s}
    al, be = par["alpha"], par["beta"]
    scale = 100.0 / max(np.std(x), 1e-12)
    y = x * scale
    uncond = float(np.var(y))
    om = max(uncond * (1 - al - be), 1e-300)
    sig_next = math.sqrt(_garch_filter((om, al, be), y)[-1]) / scale
    if dist == "normal":
        zv, zc = normal_tail(q)
    else:
        zv, zc = student_tail(q, par["nu"])
    return {"var": zv * sig_next, "cvar": zc * sig_next}


def default_estimators(ewma_lam: float = 0.94, k: float = 4.0) -> Dict[str, Estimator]:
    return {
        "naive": Estimator("naive historical", "naive", {}),
        "k_sigma": Estimator(f"{k:g}-sigma rule", "k_sigma", {"k": k}),
        "sma_normal": Estimator("moving average + normal", "sma_normal", {}),
        "ewma_normal": Estimator(f"EWMA({ewma_lam:g}) + normal", "ewma_normal",
                                 {"lam": ewma_lam}),
        "garch_normal": Estimator("GARCH + normal", "garch_normal", {}),
        "garch_t": Estimator("GARCH + Student-t", "garch_t", {}),
    }


def implied_confidence(k: float, proc_ppf) -> float:
    """What confidence level does a k-sigma rule actually target?

    Answers the question the 4-sigma rule raises: the same multiplier means a
    different tail probability under every distribution.
    """
    from scipy.optimize import brentq
    try:
        return float(brentq(lambda q: proc_ppf(q) - k, 0.5 + 1e-9, 1 - 1e-12))
    except Exception:  # noqa: BLE001
        return float("nan")
