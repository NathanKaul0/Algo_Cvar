"""
Figures for the validation half of the study.

The grid figures (mc_fig1-5) come from analyse_grid.py. These three cover the
real-data half: that the backtest is credible, what the coverage test found,
and whether the correction holds up out of sample.

Palette is the validated categorical set also used by analyse_grid.py. Two of
its hues sit below 3:1 against the surface, which obliges visible labels rather
than colour-alone identity - every series here carries a direct label.

    python make_figs.py
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from paths import data_path, fig_path                           # noqa: E402

BLUE, ORANGE, AQUA, VIOLET, RED, YELLOW = ("#2a78d6", "#eb6834", "#1baf7a",
                                           "#4a3aa7", "#e34948", "#eda100")
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#b8b7b0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "grid.color": "#e8e7e2", "grid.linewidth": 0.8, "legend.frameon": False,
})
SPLIT = pd.Timestamp("2010-01-01")


def save(fig, name):
    fig.savefig(fig_path(name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------- fig v1
def fig_equity():
    """One series, so no legend: the title names it. Drawdown below, shared x."""
    r = pd.read_csv(data_path("strategy_returns.csv"), index_col=0,
                    parse_dates=True)["ret"]
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.4, 1],
                                               "hspace": 0.12})
    ax.plot(eq.index, eq.to_numpy(), color=BLUE, lw=2.0)
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1  (log scale)")
    ax.set_title("The rebuilt backtest: 34 markets, public prices, "
                 "no transaction costs")
    ax.grid(True); ax.set_axisbelow(True)

    # the decade the industry lost, marked once rather than shaded everywhere
    ax.axvspan(pd.Timestamp("2010-01-01"), pd.Timestamp("2019-12-31"),
               color=MUTED, alpha=0.16, lw=0)
    # placed in axes fraction on the y so it cannot collide with the curve
    ax.annotate("the 2010s: Sharpe 0.30\n(industry-wide, not a bug)",
                xy=(pd.Timestamp("2014-06-01"), 0.10),
                xycoords=("data", "axes fraction"),
                color=INK2, fontsize=8.5, ha="center")
    ax.annotate(f"final {eq.iloc[-1]:.1f}x", (eq.index[-1], eq.iloc[-1]),
                xytext=(-6, 8), textcoords="offset points",
                color=BLUE, fontweight="bold", fontsize=9, ha="right")

    ax2.fill_between(dd.index, dd.to_numpy() * 100, 0, color=RED, alpha=0.5,
                     lw=0)
    ax2.plot(dd.index, dd.to_numpy() * 100, color=RED, lw=1.0)
    ax2.set_ylabel("drawdown (%)")
    ax2.grid(True); ax2.set_axisbelow(True)
    worst = dd.idxmin()
    ax2.annotate(f"worst {100*dd.min():.0f}%", (worst, dd.min() * 100),
                 xytext=(8, 4), textcoords="offset points",
                 color=RED, fontweight="bold", fontsize=8.5)
    return save(fig, "val_fig1_equity.png")


# ---------------------------------------------------------------- fig v2
def fig_coverage():
    """Breach rate against target, held-out half. Dot plot: position carries
    the value, colour only groups it, and every row is labelled."""
    import portfolio_layer as pl

    r = pd.read_csv(data_path("coverage_clean_results.csv"), index_col=0,
                    parse_dates=True)
    u = (r.loss / r["new_var_0.9"]).to_numpy(dtype=float)
    train = np.asarray(r.index < SPLIT)
    test = ~train

    rows = [("no correction", np.ones(len(r)), RED),
            ("trailing 500d  (pre-registered)", pl.k_trailing(u, 500), ORANGE),
            ("trailing 1000d", pl.k_trailing(u, 1000), BLUE)]
    levels, targets = ("0.9", "0.95", "0.99"), (10.0, 5.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    for ax, lv, tgt in zip(axes, levels, targets):
        vals = []
        for i, (nm, k, col) in enumerate(rows):
            v = 100 * float((r.loss.to_numpy()[test]
                             > (r[f"new_var_{lv}"].to_numpy() * k)[test]).mean())
            y = len(rows) - 1 - i
            ax.plot([tgt, v], [y, y], color=col, lw=2.0, alpha=0.55,
                    solid_capstyle="round", zorder=2)
            ax.plot([v], [y], "o", ms=10, color=col, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            vals.append(v)
            ax.annotate(f"{v:.2f}%", (v, y), xytext=(13, 0),
                        textcoords="offset points", color=INK,
                        fontweight="bold", fontsize=8.5, va="center")
        ax.axvline(tgt, color=INK2, lw=1.6, ls="--", zorder=1)
        ax.set_title(f"{float(lv):.0%} level   (target {tgt:.0f}%)")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([nm for nm, _, _ in rows][::-1], fontsize=8.5)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        # each panel scales to its own level, so the 99% differences are
        # visible rather than crushed against a shared 15% axis
        ax.set_xlim(0, max(max(vals), tgt) * 1.55)
        ax.set_xlabel("share of days breaching (%)")
        ax.grid(True, axis="x"); ax.set_axisbelow(True)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.suptitle("Coverage on the held-out half (2010-2024, 3,645 days): "
                 "the model's scale was 25% too small",
                 fontsize=11.5, fontweight="bold", y=1.04, x=0.02, ha="left")
    return save(fig, "val_fig2_coverage.png")


# ---------------------------------------------------------------- fig v3
def fig_rolling():
    """Rolling breach rate through time, so the fix is visible as behaviour
    rather than as one summary number."""
    import portfolio_layer as pl

    r = pd.read_csv(data_path("coverage_clean_results.csv"), index_col=0,
                    parse_dates=True)
    u = (r.loss / r["new_var_0.9"]).to_numpy(dtype=float)
    k = pl.k_trailing(u, 1000)

    W = 500
    raw = (r.loss > r["new_var_0.99"]).rolling(W).mean() * 100
    adj = pd.Series(r.loss.to_numpy() > r["new_var_0.99"].to_numpy() * k,
                    index=r.index).rolling(W).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(raw.index, raw.to_numpy(), color=RED, lw=2.0, label="no correction")
    ax.plot(adj.index, adj.to_numpy(), color=BLUE, lw=2.0,
            label="trailing 1000d correction")
    ax.axhline(1.0, color=INK2, lw=1.6, ls="--", zorder=1)
    ax.axvline(SPLIT, color=INK2, lw=1.2, ls=":", zorder=1)
    ax.annotate("target 1%", (raw.index[int(len(raw) * 0.02)], 1.0),
                xytext=(0, 6), textcoords="offset points", color=INK2,
                fontsize=9, fontweight="bold")
    ax.annotate("selection split\n<- chosen here   validated here ->",
                xy=(SPLIT, 0.88), xycoords=("data", "axes fraction"),
                color=INK2, fontsize=8.5, ha="center")
    for s, col, nm in ((raw, RED, "no correction"),
                       (adj, BLUE, "trailing 1000d")):
        v = s.dropna()
        ax.annotate(nm, (v.index[-1], v.iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", color=col, fontsize=8.5,
                    fontweight="bold", va="center")
    ax.set_ylabel(f"breach rate at 99%, rolling {W} days (%)")
    ax.set_title("The same result through time, not just in aggregate")
    ax.set_xlim(raw.index[0], raw.index[-1] + pd.Timedelta(days=1500))
    ax.grid(True); ax.set_axisbelow(True)
    # both series are direct-labelled; the legend is the required second
    # channel, parked where it cannot collide with the split marker
    ax.legend(loc="lower left", ncol=2)
    return save(fig, "val_fig3_rolling.png")


if __name__ == "__main__":
    print("building validation figures...")
    fig_equity(); fig_coverage(); fig_rolling()
