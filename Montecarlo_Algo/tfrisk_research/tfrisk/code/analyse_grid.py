"""
Step 5 - Analysis of the estimator grid.

The output is deliberately NOT a ranking. It is a failure map: for each method,
where in the parameter space it breaks and by how much. A ranking would hide
the fact that different methods fail in different places, which is the whole
point of running the grid.

Figures use the validated categorical palette; every series carries a direct
label as well as a legend, so identity is never colour-alone.
"""

from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paths import data_path, fig_path  # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = None   # figures resolve through fig_path()


BLUE, ORANGE, AQUA, VIOLET, RED, YELLOW = ("#2a78d6", "#eb6834", "#1baf7a",
                                           "#4a3aa7", "#e34948", "#eda100")
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#b8b7b0", "#fcfcfb"

NAMES = {"naive": "naive historical", "k_sigma": "4-sigma rule",
         "sma_normal": "moving average", "ewma_normal": "EWMA",
         "garch_normal": "GARCH-normal", "garch_t": "GARCH-t"}
COLOR = {"naive historical": BLUE, "4-sigma rule": MUTED,
         "moving average": ORANGE, "EWMA": YELLOW,
         "GARCH-normal": AQUA, "GARCH-t": VIOLET}
ORDER = ["naive historical", "moving average", "EWMA", "GARCH-normal",
         "GARCH-t", "4-sigma rule"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "grid.color": "#e8e7e2", "grid.linewidth": 0.8, "legend.frameon": False,
})


def load() -> pd.DataFrame:
    d = pd.read_csv(data_path("grid_results.csv"))
    d["est"] = d.estimator.map(NAMES)
    return d


def save(fig, name):
    fig.savefig(fig_path(name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def _lines(ax, piv, ylabel, xlabel, title, drop=(), pct=True, label_x=None):
    """Plot one line per estimator with a legend AND direct labels.

    Direct labels are nudged apart when series converge, so identity is never
    carried by colour alone even where two methods coincide - and two methods
    coinciding is itself a result worth being able to read off the chart.
    """
    series = []
    for est in ORDER:
        if est not in piv.index or est in drop:
            continue
        y = piv.loc[est].to_numpy(dtype=float) * (100 if pct else 1)
        x = np.arange(len(piv.columns))
        ax.plot(x, y, color=COLOR[est], lw=2.2, marker="o", ms=6, label=est)
        lx = (len(x) - 1) if label_x is None else label_x
        series.append([est, float(x[lx]), float(y[lx])])

    # de-collide the labels vertically
    if series:
        series.sort(key=lambda r: r[2])
        span = max(s[2] for s in series) - min(s[2] for s in series)
        gap = max(span * 0.055, 1e-9)
        for i in range(1, len(series)):
            if series[i][2] - series[i - 1][2] < gap:
                series[i][2] = series[i - 1][2] + gap
        side = -1 if (label_x == 0) else 1
        for est, lx, ly in series:
            ax.annotate(est, (lx, ly), xytext=(8 * side, 0),
                        textcoords="offset points", color=COLOR[est],
                        fontsize=8.5, fontweight="bold", va="center",
                        ha="left" if side > 0 else "right")

    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([f"{c:g}" for c in piv.columns])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True); ax.set_axisbelow(True)


# ---------------------------------------------------------------- fig 1
def fig_kurtosis(d):
    k = d[(d.family == "student_t") & (d.window == 500) & (d.level == 0.99)]
    piv = k.pivot_table(index="est", columns="tail_index", values="bias")
    piv = piv[sorted(piv.columns, reverse=True)]
    fig, ax = plt.subplots(figsize=(9.6, 5))
    _lines(ax, piv, "VaR error (% of truth)", "tail index  (lower = fatter tails)",
           "Fat tails: a normal quantile understates VaR, and the gap widens",
           drop=("4-sigma rule",))
    ax.axhline(0, color=INK2, lw=1.4, ls="--", zorder=1)
    ax.text(0.15, 1.2, "unbiased", color=INK2, fontsize=9, style="italic")
    ax.set_xlim(-0.3, len(piv.columns) + 1.0)
    ax.legend(loc="lower left", ncol=2)
    fig.suptitle("Window 500 days, 99% level", fontsize=10, color=INK2,
                 y=0.955, x=0.125, ha="left")
    return save(fig, "mc_fig1_kurtosis_bias.png")


# ---------------------------------------------------------------- fig 2
def fig_breach(d):
    k = d[(d.family == "student_t") & (d.window == 500) & (d.level == 0.99)]
    piv = k.pivot_table(index="est", columns="tail_index", values="breach")
    piv = piv[sorted(piv.columns, reverse=True)]
    fig, ax = plt.subplots(figsize=(9.6, 5))
    _lines(ax, piv, "share of days breaching the estimate (%)",
           "tail index  (lower = fatter tails)",
           "The operational consequence: how often losses break through",
           drop=("4-sigma rule",))
    ax.axhline(1.0, color=INK2, lw=1.8, ls="--", zorder=1)
    ax.text(0.15, 1.06, "target 1%", color=INK2, fontsize=9.5, fontweight="bold")
    ax.set_xlim(-0.3, len(piv.columns) + 1.0)
    ax.legend(loc="upper right", ncol=2)
    return save(fig, "mc_fig2_kurtosis_breach.png")


# ---------------------------------------------------------------- fig 3
def fig_clustering(d):
    g = d[(d.family == "garch") & (~np.isfinite(d.tail_index))
          & (d.window == 500) & (d.level == 0.99)]
    piv = g.pivot_table(index="est", columns="persistence", values="rmse")
    fig, ax = plt.subplots(figsize=(9.6, 5))
    _lines(ax, piv, "RMSE of the VaR estimate (% of truth)",
           "volatility persistence  (alpha + beta)",
           "Clustering: where a reactive volatility model earns its keep",
           drop=("4-sigma rule",), label_x=0)
    ax.set_xlim(-2.0, len(piv.columns) - 0.6)
    ax.legend(loc="upper left", ncol=2)
    return save(fig, "mc_fig3_clustering_rmse.png")


# ---------------------------------------------------------------- fig 4
def fig_skew(d):
    s = d[(d.family == "skew_t") & (d.window == 500) & (d.level == 0.99)]
    piv = s.pivot_table(index="est", columns="lam", values="bias")
    fig, ax = plt.subplots(figsize=(9.6, 5))
    _lines(ax, piv, "VaR error (% of truth)",
           "skew parameter   (negative = fatter LEFT tail = worse losses)",
           "Skewness: every symmetric method fails; the empirical one does not",
           drop=("4-sigma rule",))
    ax.axhline(0, color=INK2, lw=1.4, ls="--", zorder=1)
    ax.set_xlim(-0.3, len(piv.columns) + 1.1)
    ax.legend(loc="upper left", ncol=2)
    return save(fig, "mc_fig4_skew_bias.png")


# ---------------------------------------------------------------- fig 5
def fig_window(d):
    w = d[(d.process == "t(nu=5)") & (d.level == 0.99)]
    piv = w.pivot_table(index="est", columns="window", values="rmse")
    fig, ax = plt.subplots(figsize=(9.6, 5))
    _lines(ax, piv, "RMSE of the VaR estimate (% of truth)",
           "estimation window (trading days)",
           "How much history each method needs   [data: t(nu=5), 99% level]",
           drop=("4-sigma rule",))
    ax.set_xlim(-0.3, len(piv.columns) + 1.0)
    ax.legend(loc="upper right", ncol=2)
    return save(fig, "mc_fig5_window_rmse.png")


# ---------------------------------------------------------------- crossovers
def crossovers(d):
    print("\n" + "=" * 78)
    print("CROSSOVER POINTS - the numbers that justify a choice")
    print("=" * 78)

    g = d[(d.family == "garch") & (~np.isfinite(d.tail_index))
          & (d.window == 500) & (d.level == 0.99)]
    piv = g.pivot_table(index="est", columns="persistence", values="rmse")
    print("\n1. When does a reactive volatility model beat a flat window?")
    print("   RMSE, GARCH-normal vs moving average, by persistence:")
    for p in piv.columns:
        a, b = piv.loc["GARCH-normal", p], piv.loc["moving average", p]
        tag = "GARCH better" if a < b * 0.95 else ("tie" if a < b * 1.05 else "flat window better")
        print(f"     persistence {p:.2f}:  GARCH {a:.3f}  vs  moving avg {b:.3f}   -> {tag}")

    k = d[(d.family == "student_t") & (d.window == 500) & (d.level == 0.99)]
    kp = k.pivot_table(index="est", columns="tail_index", values="breach")
    print("\n2. When does assuming a normal tail stop being acceptable?")
    print("   breach rate against a 1% target, by tail index:")
    for t in sorted(kp.columns, reverse=True):
        v = kp.loc["GARCH-normal", t]
        tag = "ok" if v < 0.0125 else ("marginal" if v < 0.015 else "FAILS")
        print(f"     tail index {t:4.1f}:  {100*v:.2f}%   -> {tag}")

    w = d[(d.process == "t(nu=5)") & (d.level == 0.99)]
    wp = w.pivot_table(index="est", columns="window", values="rmse")
    print("\n3. How much history before the empirical method overtakes the parametric one?")
    for W in wp.columns:
        a, b = wp.loc["naive historical", W], wp.loc["GARCH-t", W]
        tag = "naive better" if a < b else "GARCH-t better"
        print(f"     window {W:5d}:  naive {a:.3f}  vs  GARCH-t {b:.3f}   -> {tag}")

    print("\n4. The cost of over-fitting: GARCH-t on data with NO fat tails and")
    print("   NO clustering (the normal control), by window:")
    c = d[(d.family == "normal") & (d.level == 0.99)]
    cp = c.pivot_table(index="est", columns="window", values="rmse")
    for W in cp.columns:
        print(f"     window {W:5d}:  GARCH-t {cp.loc['GARCH-t', W]:.3f}  vs  "
              f"moving avg {cp.loc['moving average', W]:.3f}  "
              f"(penalty {100*(cp.loc['GARCH-t',W]/cp.loc['moving average',W]-1):+.0f}%)")

    s = d[(d.family == "skew_t") & (d.window == 500) & (d.level == 0.99)]
    sp = s.pivot_table(index="est", columns="lam", values="bias")
    print("\n5. Skewness: VaR error at the extremes of the skew dial")
    for est in ("naive historical", "GARCH-t", "moving average"):
        print(f"     {est:<18} lam=-0.6: {100*sp.loc[est,-0.6]:+6.1f}%   "
              f"lam=+0.6: {100*sp.loc[est,0.6]:+6.1f}%")


if __name__ == "__main__":
    d = load()
    print(f"loaded {len(d)} rows, {d.process.nunique()} processes, "
          f"{d.window.nunique()} windows, {d.level.nunique()} levels")
    print("\nbuilding figures...")
    fig_kurtosis(d); fig_breach(d); fig_clustering(d); fig_skew(d); fig_window(d)
    crossovers(d)
