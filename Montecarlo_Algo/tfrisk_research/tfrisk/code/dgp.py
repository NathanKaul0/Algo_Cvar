"""
Step 2 - Data generating processes for the synthetic VaR study.

DESIGN RULES
------------
1. ONE PROPERTY AT A TIME. Fat tails and volatility clustering both raise
   kurtosis. If both are varied together nothing can be attributed, so each
   family isolates a single dial:

       normal    -> the control. No skew, no excess kurtosis, no clustering.
       student_t -> kurtosis only  (dial: nu)
       skew_t    -> skewness       (dial: lam, with nu held fixed)
       garch     -> clustering     (dial: alpha+beta, innovations normal or t)

2. UNIT UNCONDITIONAL VARIANCE for every process at every parameter setting.
   Without this, a "fatter tails" setting also changes the scale and the VaR
   comparison confounds shape with size. Every family below is standardised
   analytically, not by dividing out a sample standard deviation (which would
   be a different, data-dependent object).

3. The skewed-t is Hansen (1994), which is defined to have zero mean and unit
   variance by construction and has a closed-form quantile function - so the
   skew family gets EXACT ground truth rather than a simulated approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------------
# Hansen (1994) skewed t
# ---------------------------------------------------------------------------

def _hansen_abc(eta: float, lam: float) -> Tuple[float, float, float]:
    """Constants a, b, c for Hansen's skewed t. Requires eta > 2, |lam| < 1."""
    c = math.gamma((eta + 1) / 2) / (math.sqrt(math.pi * (eta - 2)) * math.gamma(eta / 2))
    a = 4 * lam * c * ((eta - 2) / (eta - 1))
    b = math.sqrt(1 + 3 * lam ** 2 - a ** 2)
    return a, b, c


def hansen_ppf(u, eta: float, lam: float):
    """Quantile function of Hansen's skewed t (zero mean, unit variance).

    Piecewise inverse: below the kink the left scale applies, above it the
    right scale. Setting lam = 0 reduces exactly to a standardised Student-t.
    """
    u = np.asarray(u, dtype=float)
    a, b, _ = _hansen_abc(eta, lam)
    s = math.sqrt((eta - 2) / eta)
    cut = (1 - lam) / 2
    out = np.empty_like(u)

    lo = u < cut
    if np.any(lo):
        inner = u[lo] / (1 - lam)
        out[lo] = (1 - lam) / b * s * sps.t.ppf(inner, eta) - a / b
    hi = ~lo
    if np.any(hi):
        inner = 0.5 + (u[hi] - cut) / (1 + lam)
        out[hi] = (1 + lam) / b * s * sps.t.ppf(inner, eta) - a / b
    return out


def hansen_pdf(z, eta: float, lam: float):
    z = np.asarray(z, dtype=float)
    a, b, c = _hansen_abc(eta, lam)
    bz = b * z + a
    denom = np.where(z < -a / b, 1 - lam, 1 + lam)
    return b * c * (1 + (1 / (eta - 2)) * (bz / denom) ** 2) ** (-(eta + 1) / 2)


def hansen_rvs(n: int, eta: float, lam: float, rng: np.random.Generator):
    return hansen_ppf(rng.random(n), eta, lam)


# ---------------------------------------------------------------------------
# Process specification
# ---------------------------------------------------------------------------

def _skew_q(nu: float, lam: float) -> float:
    from scipy import integrate as _it
    return float(_it.quad(lambda z: z ** 3 * hansen_pdf(z, nu, lam), -200, 200,
                          limit=800)[0])


@dataclass
class Process:
    """A data generating process, standardised to unit unconditional variance."""
    family: str                       # normal | student_t | skew_t | garch
    params: Dict[str, float] = field(default_factory=dict)

    # ---- identification -------------------------------------------------
    @property
    def name(self) -> str:
        p = self.params
        if self.family == "normal":
            return "normal"
        if self.family == "student_t":
            return f"t(nu={p['nu']:g})"
        if self.family == "skew_t":
            return f"skew-t(nu={p['nu']:g},lam={p['lam']:+g})"
        if self.family == "garch":
            innov = p.get("innov", "normal")
            if innov == "normal":
                tag = "normal"
            elif innov == "skew_t":
                tag = f"skt{p.get('nu', 7):g}/{p.get('lam', 0.0):+g}"
            else:
                tag = f"t{p.get('nu', 7):g}"
            return f"garch(a={p['alpha']:g},b={p['beta']:g},{tag})"
        raise ValueError(self.family)

    @property
    def is_iid(self) -> bool:
        return self.family != "garch"

    @property
    def tail_index(self) -> float:
        """Degrees of freedom of the innovation law; inf for Gaussian tails.

        This, NOT kurtosis, is the right x-axis for the study. Kurtosis is
        undefined once nu <= 4 -- which includes markets observed in the real
        data (Phase 1 found per-market nu from 2.7 to 18.9) -- and even where
        defined it is dominated by a handful of observations. The tail index is
        always defined and is the parameter actually being varied.
        """
        p = self.params
        if self.family == "normal":
            return float("inf")
        if self.family in ("student_t", "skew_t"):
            return float(p["nu"])
        if self.family == "garch":
            return (float("inf") if p.get("innov", "normal") == "normal"
                    else float(p.get("nu", 7.0)))
        return float("nan")

    @property
    def has_finite_kurtosis(self) -> bool:
        e = self.moments()["exkurt"]
        return bool(math.isfinite(e))

    @property
    def is_variance_stationary(self) -> bool:
        """Unconditional variance exists and equals 1. This is the property the
        study genuinely requires; a finite 4th moment is not required."""
        p = self.params
        if self.family == "garch":
            return p["alpha"] + p["beta"] < 1.0
        if self.family in ("student_t", "skew_t"):
            return p["nu"] > 2.0
        return True

    # ---- innovation distribution (unit variance) -------------------------
    def _innov_rvs(self, n: int, rng: np.random.Generator) -> np.ndarray:
        p = self.params
        if self.family == "normal":
            return rng.standard_normal(n)
        if self.family == "student_t":
            nu = p["nu"]
            return rng.standard_t(nu, n) * math.sqrt((nu - 2) / nu)
        if self.family == "skew_t":
            return hansen_rvs(n, p["nu"], p["lam"], rng)
        if self.family == "garch":
            iv = p.get("innov", "normal")
            if iv == "normal":
                return rng.standard_normal(n)
            if iv == "skew_t":
                # the combination the first grid lacked: clustering AND skew.
                # Real data has both, and the two point to different estimators.
                return hansen_rvs(n, p.get("nu", 7.0), p.get("lam", 0.0), rng)
            nu = p.get("nu", 7.0)
            return rng.standard_t(nu, n) * math.sqrt((nu - 2) / nu)
        raise ValueError(self.family)

    def innov_ppf(self, q: float) -> float:
        """Quantile of the (unit-variance) innovation distribution."""
        p = self.params
        if self.family == "garch" and p.get("innov") == "skew_t":
            return float(hansen_ppf(np.array([q]), p.get("nu", 7.0),
                                    p.get("lam", 0.0))[0])
        if self.family in ("normal",) or (self.family == "garch"
                                          and p.get("innov", "normal") == "normal"):
            return float(sps.norm.ppf(q))
        if self.family == "student_t" or self.family == "garch":
            nu = p.get("nu", 7.0)
            return float(sps.t.ppf(q, nu) * math.sqrt((nu - 2) / nu))
        if self.family == "skew_t":
            return float(hansen_ppf(np.array([q]), p["nu"], p["lam"])[0])
        raise ValueError(self.family)

    # ---- simulation ------------------------------------------------------
    def simulate(self, n: int, rng: np.random.Generator,
                 burn: int = 2000, return_sigma: bool = False):
        """Generate n observations with unit unconditional variance.

        For GARCH a burn-in is discarded so the process starts from its
        stationary distribution rather than from an arbitrary seed; omega is
        set to 1-alpha-beta so the unconditional variance is exactly 1.
        """
        if self.is_iid:
            x = self._innov_rvs(n, rng)
            return (x, np.ones(n)) if return_sigma else x

        p = self.params
        alpha, beta = p["alpha"], p["beta"]
        if alpha + beta >= 1:
            raise ValueError("alpha+beta must be < 1 for a stationary variance")
        omega = 1.0 - alpha - beta
        z = self._innov_rvs(n + burn, rng)
        var = 1.0
        out = np.empty(n + burn)
        sig = np.empty(n + burn)
        for t in range(n + burn):
            sig[t] = math.sqrt(var)
            out[t] = sig[t] * z[t]
            var = omega + alpha * out[t] ** 2 + beta * var
        x, s = out[burn:], sig[burn:]
        return (x, s) if return_sigma else x

    # ---- theoretical moments where they exist ---------------------------
    def moments(self) -> Dict[str, float]:
        """Population skewness and excess kurtosis (nan where undefined)."""
        p = self.params
        if self.family == "normal":
            return {"skew": 0.0, "exkurt": 0.0}
        if self.family == "student_t":
            nu = p["nu"]
            return {"skew": 0.0 if nu > 3 else float("nan"),
                    "exkurt": 6 / (nu - 4) if nu > 4 else float("inf")}
        if self.family == "skew_t":
            # Hansen's moments have no tidy closed form; quadrature to machine
            # tolerance is exact enough and is independently checkable.
            from scipy import integrate as _it
            nu, lam = p["nu"], p["lam"]
            if nu <= 4:
                return {"skew": float("nan") if nu <= 3 else _skew_q(nu, lam),
                        "exkurt": float("inf")}
            m3 = _it.quad(lambda z: z ** 3 * hansen_pdf(z, nu, lam), -200, 200,
                          limit=800)[0]
            m4 = _it.quad(lambda z: z ** 4 * hansen_pdf(z, nu, lam), -200, 200,
                          limit=800)[0]
            return {"skew": float(m3), "exkurt": float(m4 - 3.0)}
        if self.family == "garch":
            a, b = p["alpha"], p["beta"]
            if p.get("innov", "normal") == "skew_t":
                sub = Process("skew_t", {"nu": p.get("nu", 7.0),
                                         "lam": p.get("lam", 0.0)})
                k_z = sub.moments()["exkurt"] + 3.0
            elif p.get("innov", "normal") == "normal":
                k_z = 3.0
            else:
                nu = p.get("nu", 7.0)
                k_z = 3 * (nu - 2) / (nu - 4) if nu > 4 else float("inf")
            # Bollerslev (1986): finite 4th moment requires k_z*a^2 + 2ab + b^2 < 1
            cond = k_z * a ** 2 + 2 * a * b + b ** 2
            if cond >= 1 or not math.isfinite(k_z):
                return {"skew": 0.0, "exkurt": float("inf")}
            kurt = k_z * (1 - (a + b) ** 2) / (1 - cond)
            return {"skew": 0.0, "exkurt": kurt - 3.0}
        raise ValueError(self.family)


# ---------------------------------------------------------------------------
# The experiment grid
# ---------------------------------------------------------------------------

def default_grid() -> Dict[str, list]:
    """Processes for the study, grouped by which property they isolate."""
    return {
        "control": [Process("normal")],
        "kurtosis": [Process("student_t", {"nu": nu})
                     for nu in (30, 15, 10, 8, 6, 5, 4, 3.5)],
        "skewness": [Process("skew_t", {"nu": 8, "lam": lam})
                     for lam in (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6)],
        "clustering": [Process("garch", {"alpha": a, "beta": b, "innov": "normal"})
                       for a, b in ((0.03, 0.77), (0.05, 0.85), (0.08, 0.88),
                                    (0.10, 0.88), (0.12, 0.86), (0.06, 0.93))],
        "clustering_fat": [Process("garch", {"alpha": a, "beta": b,
                                             "innov": "t", "nu": 7})
                           for a, b in ((0.05, 0.85), (0.08, 0.88), (0.10, 0.88))],
        # Step 6: cells bracketing the parameters MEASURED on the real book --
        # tail index ~6-7, persistence ~0.984, residual skew ~-0.21. This is the
        # combination the first grid lacked, and the one where the fat-tail and
        # the skew findings give conflicting advice.
        "realistic": [
            Process("garch", {"alpha": 0.08, "beta": 0.904, "innov": "skew_t",
                              "nu": nu, "lam": lam})
            for nu, lam in ((7.0, 0.00), (7.0, -0.08), (7.0, -0.20),
                            (6.0, -0.08), (10.0, -0.08))
        ],
    }
