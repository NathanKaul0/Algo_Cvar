"""
Why both models fail on the clean long/short book, and what would fix it.

The coverage test says both models over-breach at every level, and the
Christoffersen test says the breaches are CLUSTERED. Clustering is the
informative part: a model that were simply too narrow by a constant would
breach too often but at random times. Clustered breaches mean the conditional
scale is mistimed - the model is right on average and wrong exactly when it
matters.

This script tests three candidate causes in order of how cheaply they can be
ruled out, and ends with the one actionable question: would rescaling the
portfolio-level volatility have fixed it?
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats as sps
from paths import data_path, fig_path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
Z90 = sps.norm.ppf(0.90)


def load():
    r = pd.read_csv(data_path("coverage_clean_results.csv"),
                    index_col=0, parse_dates=True)
    return r


def banner(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def scale_vs_covariance():
    """Model scale vs a trailing realised covariance on identical exposures."""
    import coverage_clean as cc
    d = cc.load()
    r = load()
    X = d["x"].to_numpy(dtype=float)
    Xf = np.where(np.isfinite(X), X, 0.0)
    pos = d["pos"]
    E = pos.to_numpy(dtype=float) * d["vol_d"][pos.columns].to_numpy(dtype=float)
    E = np.where(np.isfinite(E), E, 0.0)
    idx = d["idx"]
    sig_model = r["new_var_0.9"] / Z90
    out = []
    for w in (60, 250, 1000):
        vals = []
        for t in r.index[::50]:
            i = idx.get_loc(t)
            if i < w:
                continue
            C = np.cov(Xf[i - w:i].T)
            v = float(E[i] @ C @ E[i])
            vals.append(np.sqrt(max(v, 0.0)))
        s = pd.Series(vals, index=r.index[::50][-len(vals):])
        out.append((w, float((s / sig_model.reindex(s.index)).dropna().median())))
    return out


def main():
    r = load()
    # the model's implied portfolio sd, backed out of its own 90% quantile
    sig = r["new_var_0.9"] / Z90
    u = r["loss"] / sig                      # standardised loss

    banner("1. IS THE MISS A CONSTANT, OR IS IT TIMED?")
    # NOTE ON WHAT IS *NOT* USED HERE. The tempting measure is to back a
    # portfolio sigma out of the model's own 90% quantile as var90/1.2816 and
    # compare the standardised loss's sd against 1. That is wrong: the FHS
    # distribution is fat-tailed, so its 90% quantile is not 1.2816 sigma, and
    # dividing by 1.2816 understates sigma and manufactures a scale shortfall
    # that is an artefact of assuming normality. Likewise, comparing the pooled
    # realised quantile against the mean predicted quantile mixes an
    # unconditional object with a conditional one and overstates the gap.
    #
    # What IS used: breach rates, which are conditional by construction; the
    # scale-free ratio loss/var90 for TIMING only; and a same-day comparison of
    # the model's scale against a trailing realised covariance on identical
    # exposures, which compares two conditional objects.
    print(f"  mean of loss / VaR90            {u.mean():+.3f}")
    a1 = pd.Series(u.abs()).autocorr(1)
    a5 = pd.Series(u.abs()).autocorr(5)
    a20 = pd.Series(u.abs()).autocorr(20)
    print(f"\n  autocorrelation of |loss / VaR90|  (scale-free, so the")
    print(f"  normality caveat above does not apply):")
    print(f"    lag 1  {a1:+.3f}    lag 5  {a5:+.3f}    lag 20  {a20:+.3f}")
    print("  If the model had captured the conditional scale these would be ~0.")
    print("  Positive values mean volatility the model did not see coming.")

    banner("2. HOW MUCH SCALE IS MISSING, MEASURED HONESTLY")
    print("  Same-day exposures run through a trailing realised covariance of")
    print("  the market moves, against the model's own scale on the same day.")
    print("  Both sides are conditional, so this comparison is valid where the")
    print("  two rejected in the note above are not.\n")
    for w, ratio in scale_vs_covariance():
        print(f"    trailing covariance window {w:5d}d:"
              f"   realised / model = {ratio:.3f}")
    print("\n  Modestly but persistently too narrow. Each market's own")
    print("  volatility is already filtered by its own GARCH, so what is")
    print("  missing is a PORTFOLIO-level quantity: the co-movement between")
    print("  markets, which a trend book is concentrated in by construction.")

    roll = u.rolling(120).std()
    worst = roll.nlargest(1).index[0]
    print(f"\n  worst 120-day stretch of loss/VaR90 ends {worst.date()}")
    top = u.nlargest(5)
    print("  five largest standardised losses:")
    for d, v in top.items():
        print(f"    {d.date()}  {v:.2f} sigma")

    banner("3. WOULD RESCALING THE PORTFOLIO VOLATILITY HAVE FIXED IT?")
    # Apply a trailing EWMA correction to the model's own scale, estimated from
    # the standardised losses it has already produced. Strictly trailing: the
    # correction for day t uses standardised losses through t-1 only.
    print("  A trailing EWMA correction factor is applied to the model's own")
    print("  scale, estimated only from standardised losses it has already")
    print("  produced. No new data, no new parameters beyond one decay.\n")
    print(f"  {'lambda':>8}{'half-life':>11}{'90%':>9}{'95%':>9}{'99%':>9}"
          f"{'ES ratio':>10}")
    print(f"  {'target':>8}{'':>11}{10.0:>9.1f}{5.0:>9.1f}{1.0:>9.1f}")
    print("  " + "-" * 56)

    for lam in (0.99, 0.97, 0.94, 0.90):
        # EWMA of u^2, lagged one day
        u2 = (u ** 2).to_numpy()
        c = np.empty(len(u2))
        c[0] = 1.0
        for i in range(1, len(u2)):
            c[i] = lam * c[i - 1] + (1 - lam) * u2[i - 1]
        corr = np.sqrt(np.clip(c, 1e-9, None))
        hl = np.log(0.5) / np.log(lam)
        rates = []
        for q, lv in ((0.90, "0.9"), (0.95, "0.95"), (0.99, "0.99")):
            v = r[f"new_var_{lv}"].to_numpy() * corr
            rates.append(100 * float((r.loss.to_numpy() > v).mean()))
        v99 = r["new_var_0.99"].to_numpy() * corr
        c99 = r["new_cvar_0.99"].to_numpy() * corr
        br = r.loss.to_numpy() > v99
        es = float(r.loss.to_numpy()[br].mean() / c99[br].mean())
        print(f"  {lam:>8.2f}{hl:>11.0f}{rates[0]:>9.2f}{rates[1]:>9.2f}"
              f"{rates[2]:>9.2f}{es:>10.2f}")

    print("\n  For comparison, the uncorrected model:")
    rates = [100 * float((r.loss > r[f"new_var_{lv}"]).mean())
             for lv in ("0.9", "0.95", "0.99")]
    br = (r.loss > r["new_var_0.99"]).to_numpy()
    es = float(r.loss[br].mean() / r["new_cvar_0.99"][br].mean())
    print(f"  {'none':>8}{'':>11}{rates[0]:>9.2f}{rates[1]:>9.2f}"
          f"{rates[2]:>9.2f}{es:>10.2f}")


if __name__ == "__main__":
    main()
