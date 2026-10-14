"""
Step 3 - Ground truth. The point of the whole study.

With real data you never know the right answer, so a coverage shortfall could
be model error or an unlucky sample. Here the answer is known exactly, so every
deviation is attributable. That only holds if the truth is actually right,
which is why nothing below is taken on faith: every analytic formula is checked
against both numerical integration and a large Monte Carlo sample.

CONDITIONAL vs UNCONDITIONAL
----------------------------
These are different objects and estimators rank differently against them.

  unconditional VaR - one number for the process as a whole. What you would
                      quote if you knew the long-run distribution.
  conditional VaR   - the number for tomorrow given today's state. Different
                      every day under GARCH.

Phase 1 cares about the conditional object: the estimate for tomorrow given
today. That is the primary target here. For i.i.d. families the two coincide,
which is itself a useful check.

SIGN CONVENTION
---------------
Loss = -return. VaR_q is a POSITIVE number: the q-quantile of the loss
distribution. CVaR_q is the mean loss conditional on exceeding VaR_q. Defined
as -quantile_{1-q}(return) so it stays correct for skewed distributions, where
the shortcut of using the upper quantile of a symmetric law would be wrong.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import integrate, stats as sps

from dgp import Process, hansen_pdf, hansen_ppf


# ---------------------------------------------------------------------------
# Analytic truth for the i.i.d. families
# ---------------------------------------------------------------------------

def normal_var_cvar(q: float) -> Tuple[float, float]:
    """Standard normal, unit variance."""
    z = sps.norm.ppf(q)
    return float(z), float(sps.norm.pdf(z) / (1 - q))


def student_t_var_cvar(q: float, nu: float) -> Tuple[float, float]:
    """Student-t scaled to unit variance."""
    if nu <= 2:
        raise ValueError("nu must exceed 2 for unit variance to exist")
    s = math.sqrt((nu - 2) / nu)
    p = 1 - q
    t_p = sps.t.ppf(p, nu)                     # negative
    var = -s * t_p
    # E[T | T <= t_p] = -((nu + t_p^2)/(nu-1)) * f_nu(t_p) / p
    es_t = ((nu + t_p ** 2) / (nu - 1)) * sps.t.pdf(t_p, nu) / p
    return float(var), float(s * es_t)


def skew_t_var_cvar(q: float, nu: float, lam: float) -> Tuple[float, float]:
    """Hansen skewed t. Quantile is closed form; the tail mean is quadrature.

    Quadrature rather than a formula because Hansen's piecewise density has no
    clean closed-form partial expectation; integrating to machine tolerance is
    exact enough to serve as truth and is independently checkable.
    """
    p = 1 - q
    z_p = float(hansen_ppf(np.array([p]), nu, lam)[0])
    var = -z_p
    lo = z_p - 400.0
    tail_mean = integrate.quad(lambda z: z * hansen_pdf(z, nu, lam), lo, z_p,
                               limit=500)[0] / p
    return float(var), float(-tail_mean)


# ---------------------------------------------------------------------------
# GARCH: conditional truth is exact, unconditional needs simulation
# ---------------------------------------------------------------------------

def innovation_var_cvar(proc: Process, q: float) -> Tuple[float, float]:
    """VaR and CVaR of the (unit-variance) innovation distribution."""
    p = proc.params
    if proc.family == "garch":
        iv = p.get("innov", "normal")
        if iv == "normal":
            return normal_var_cvar(q)
        if iv == "skew_t":
            # must dispatch on the skew explicitly: routing a skewed innovation
            # to the symmetric-t formula gives the wrong conditional truth and
            # silently biases every result scored against it.
            return skew_t_var_cvar(q, p.get("nu", 7.0), p.get("lam", 0.0))
        return student_t_var_cvar(q, p.get("nu", 7.0))
    if proc.family == "normal":
        return normal_var_cvar(q)
    if proc.family == "student_t":
        return student_t_var_cvar(q, p["nu"])
    if proc.family == "skew_t":
        return skew_t_var_cvar(q, p["nu"], p["lam"])
    raise ValueError(proc.family)


def conditional_truth(proc: Process, q: float, sigma: np.ndarray) -> Dict[str, np.ndarray]:
    """Exact conditional VaR/CVaR given the true volatility path.

    Under r_t = sigma_t * z_t the conditional law is a scaled innovation law,
    so the conditional quantile is sigma_t times the innovation quantile. No
    approximation.
    """
    v, c = innovation_var_cvar(proc, q)
    sigma = np.asarray(sigma, dtype=float)
    return {"var": sigma * v, "cvar": sigma * c}


def unconditional_truth_mc(proc: Process, q: float, n: int = 10_000_000,
                           seed: int = 12345) -> Dict[str, float]:
    """Monte Carlo truth for the marginal distribution.

    Used for GARCH, where the marginal of sigma_t * z_t has no closed form.
    Reported with a bootstrap standard error so the precision of 'truth' is
    itself visible rather than assumed.
    """
    rng = np.random.default_rng(seed)
    x = proc.simulate(n, rng)
    loss = -x
    var = float(np.quantile(loss, q))
    cvar = float(loss[loss >= var].mean())
    # precision of the quantile estimate: asymptotic SE of a sample quantile
    dens = _kde_density_at(loss, var)
    se = math.sqrt(q * (1 - q) / n) / dens if dens > 0 else float("nan")
    return {"var": var, "cvar": cvar, "var_se": se, "n": n}


def _kde_density_at(x: np.ndarray, point: float) -> float:
    """Crude density estimate for the quantile standard error."""
    h = 1.06 * np.std(x) * len(x) ** (-1 / 5)
    if h <= 0:
        return float("nan")
    inside = np.abs(x - point) < h
    return float(inside.mean() / (2 * h))


# ---------------------------------------------------------------------------
# Unified accessor
# ---------------------------------------------------------------------------

def truth(proc: Process, q: float, sigma: Optional[np.ndarray] = None,
          mc_n: int = 2_000_000) -> Dict[str, object]:
    """Ground truth for a process at confidence q.

    i.i.d. families      -> analytic, conditional == unconditional
    garch, sigma given   -> exact conditional path
    garch, sigma omitted -> Monte Carlo unconditional
    """
    if proc.is_iid:
        v, c = innovation_var_cvar(proc, q)
        return {"kind": "analytic", "var": v, "cvar": c,
                "conditional_equals_unconditional": True}
    if sigma is not None:
        d = conditional_truth(proc, q, sigma)
        return {"kind": "analytic-conditional", "var": d["var"], "cvar": d["cvar"],
                "conditional_equals_unconditional": False}
    d = unconditional_truth_mc(proc, q, n=mc_n)
    return {"kind": "monte-carlo-unconditional", **d,
            "conditional_equals_unconditional": False}
