#!/usr/bin/env python3
"""
make_figures.py — the five figures.
Larry (Peilin) Zhong

Fig 1  the null of the prominence statistic
Fig 2  synchrony defeats the surrogate null
Fig 3  the harmonic comb, and the notch
Fig 4  MAIN: period vs tau_GABA -- the pre-registered PING test
Fig 5  screens cannot rescue a rhythm the statistic never found

Figures 1, 2, 3 and 5 are recomputed from scratch on every run. Figure 4 plots the
network sweep in ping_scaling_test.py, which takes 25 minutes; its values are read
from that script's own output and are listed in the source below.

Writes to ../figures/. That is the only place the figures live.
Output: figure1..figure5 as .pdf (vector, for the manuscript) and .png (preview).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import sys
sys.path.insert(0, ".")
from spectral_null import (pop_rate, psd, prominence, prominence_notched, DT_S,
                           print_version_banner)
import artifact_demo as A

# ---- house style: grayscale, no chartjunk -----------------------------------
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "font.family": "sans-serif", "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
})
K, G = "#000000", "#808080"
LG = "#C8C8C8"

# ONE legend rule, everywhere: outside the axes, above, horizontal. A legend that
# is placed "where it fits" is placed by a judgement call once per figure, and
# collides with the data.
def legend_above(ax, ncol=2, y=1.02):
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, y),
              ncol=ncol, fontsize=7.5, handlelength=1.8, columnspacing=1.4,
              borderaxespad=0)


def panel(ax, letter):
    """Panel letter only. No axes titles: they occupy the same strip as the
    legend. Descriptions go in the caption."""
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def save(fig, name):
    """Write to ../figures/. That is the only copy of the figures."""
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote ../figures/{name}.pdf / .png")


# =============================================================================
# FIG 1 — the null of the statistic. Recomputed live.
# =============================================================================
def figure1():
    print("figure 1: null of the prominence statistic (recomputing, ~30 s)")
    vals = []
    for k in range(200):
        st = A.poisson_spikes(A.BASE_HZ, rng=np.random.default_rng(1000 + k))
        r = pop_rate(st, A.N_NEURON, A.T_S)
        f, P = psd(r, 1.0 / DT_S)
        vals.append(prominence(f, P, A.GAMMA)[0])
    v = np.array(vals)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    fig.subplots_adjust(top=0.74)
    ax.hist(v, bins=28, color=LG, edgecolor=K, linewidth=0.6,
            label="200 datasets with NO rhythm")

    # Every marker goes in the legend; no inline annotation on a crowded axis.
    p95 = np.percentile(v, 95)
    ax.axvline(0, color=K, lw=1.4, ls="-",
               label="0 dB: what an unbiased statistic would give")
    ax.axvline(p95, color=K, lw=1.1, ls="--",
               label=f"95th percentile of the null: {p95:.2f} dB")
    ax.axvline(6.9, color=G, lw=1.1, ls=":",
               label="v1 reported theta at 6.6–7.5 dB")
    ax.axvline(11.5, color=G, lw=1.1, ls="-.",
               label="v1 reported gamma at 11.3–11.9 dB")

    ax.set_xlabel("peak prominence (dB)")
    ax.set_ylabel("count")
    ax.set_xlim(-0.6, 13.5)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02),
              ncol=1, fontsize=7.2, handlelength=2.2, borderaxespad=0,
              labelspacing=0.35)
    save(fig, "figure1_null_of_statistic")
    return v


# =============================================================================
# FIG 2 — synchrony defeats the null; the notch fixes it.
# Recomputed from artifact_demo's E4, not transcribed from its log.
# =============================================================================
def figure2():
    print("figure 2: synchrony defeats the surrogate null (recomputing, ~3 min)")
    jit = np.array(A.JITTER_SWEEP, float)
    raw_prom, notch_prom, raw_fp, notch_fp = [], [], [], []
    for j in jit:
        raw, ntc = [], []
        for k in range(A.N_REP_SWEEP):
            st = A.synchronous_spikes(A.BASE_HZ, jitter_ms=j, depth_gamma=0.0,
                                      rng=np.random.default_rng(4000 + k))
            for notch, acc in ((False, raw), (True, ntc)):
                acc.append(A.test_rhythm(st, A.N_NEURON, A.T_S, band=A.GAMMA,
                                         n_surr=A.N_SURR_SWEEP, method="jitter",
                                         seed=k, drive_hz=A.DRIVE_HZ, notch=notch))
        raw_prom.append(np.mean([r["obs_db"] for r in raw]))
        notch_prom.append(np.mean([r["obs_db"] for r in ntc]))
        raw_fp.append(100 * np.mean([r["significant"] for r in raw]))
        notch_fp.append(100 * np.mean([r["significant"] for r in ntc]))
    raw_prom = np.array(raw_prom); notch_prom = np.array(notch_prom)
    raw_fp = np.array(raw_fp); notch_fp = np.array(notch_fp)
    null_mean = 5.62

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    fig.subplots_adjust(wspace=0.30, top=0.80)

    # categorical x: the tau values are not evenly spaced, and a numeric axis
    # crushes the interesting end of the sweep into the corner.
    xi = np.arange(len(jit))
    a1.plot(xi, raw_prom, "o-", color=K, lw=1.4, ms=5, label="raw statistic")
    a1.plot(xi, notch_prom, "s--", color=G, lw=1.4, ms=4.5, label="comb-notched")
    a1.axhline(null_mean, color=K, lw=0.9, ls=":")
    a1.text(len(jit) - 0.1, null_mean + 1.6, "null of the statistic",
            fontsize=7.5, ha="right")
    a1.set_xticks(xi); a1.set_xticklabels([f"{j:.0f}" for j in jit])
    a1.set_xlabel("theta phase-locking jitter σ (ms)   →  tighter")
    a1.set_ylabel("gamma prominence (dB)")
    a1.set_ylim(0, 41)
    legend_above(a1)
    panel(a1, "A")

    x = np.arange(len(jit)); w = 0.36
    a2.bar(x - w/2, raw_fp, w, color="#404040", edgecolor=K, lw=0.6,
           label="raw + surrogate null")
    a2.bar(x + w/2, notch_fp, w, color="white", edgecolor=K, lw=0.6, hatch="///",
           label="comb-notched + null")
    a2.axhline(5, color=K, lw=0.9, ls=":")
    # x = len(jit)-0.45 put this INSIDE the rightmost bar, where it was invisible.
    # There is no free space among the bars, so it goes in the left margin.
    a2.text(-0.5, 5, "α = 5%", fontsize=7.5, ha="left", va="bottom")
    a2.set_xticks(x); a2.set_xticklabels([f"{j:.0f}" for j in jit])
    a2.set_xlabel("theta phase-locking jitter σ (ms)")
    a2.set_ylabel("datasets called SIGNIFICANT (%)")
    legend_above(a2)
    a2.set_ylim(0, 112)
    a2.set_xlim(-0.6, len(jit) - 0.4)
    panel(a2, "B")

    save(fig, "figure2_synchrony_defeats_null")


# =============================================================================
# FIG 3 — the harmonic comb, and what the notch does. Recomputed live.
# =============================================================================
def figure3_inline():
    print("figure 3: the harmonic comb and the notch (recomputing, ~20 s)")

    def volleys(f_hz, jit_ms, n_per, T, rng, phase=0.25):
        n_cyc = int(T * f_hz)
        st = [((c + phase) / f_hz) + rng.normal(0, jit_ms / 1000.0, n_per)
              for c in range(n_cyc)]
        st = np.concatenate(st)
        return st[(st >= 0) & (st < T)]

    def gt(T, rng, gamma_hz=None, frac=0.5, n_total=A.N_NEURON * A.BASE_HZ):
        n_th = int(round(n_total * ((1 - frac) if gamma_hz else 1.0) / 6.0))
        st = [volleys(6.0, 5.0, max(n_th, 1), T, rng)]
        if gamma_hz:
            st.append(volleys(gamma_hz, 3.0,
                              max(int(round(n_total * frac / gamma_hz)), 1),
                              T, rng, phase=0.5))
        return np.sort(np.concatenate(st))

    T = A.T_S
    rng = np.random.default_rng(600)
    st_no = gt(T, np.random.default_rng(600))            # comb only, NO gamma
    st_40 = gt(T, np.random.default_rng(600), gamma_hz=40.0)  # comb + REAL 40 Hz

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True)
    fig.subplots_adjust(hspace=0.42)

    for ax, st, letter in [(a1, st_no, "A"), (a2, st_40, "B")]:
        r = pop_rate(st, A.N_NEURON, T)
        f, P = psd(r, 1.0 / DT_S)
        m = (f >= 20) & (f <= 90)
        bg = np.median(P[m])
        ax.plot(f[m], 10 * np.log10(P[m] / bg), color=K, lw=0.7)
        for k in range(4, 15):
            h = 6 * k
            if 20 <= h <= 90:
                ax.axvspan(h - 0.5, h + 0.5, color=LG, alpha=0.75, lw=0)
        ax.axvspan(30, 80, color="none", ec=G, lw=0.8, ls="--")
        p_raw, f_raw, _ = prominence(f, P, (30, 80))
        p_n, f_n, _ = prominence_notched(f, P, (30, 80), 6.0)
        ax.plot([f_raw], [10 * np.log10(P[np.argmin(abs(f - f_raw))] / bg)],
                "v", color=K, ms=8, label=f"raw max: {f_raw:.1f} Hz ({p_raw:.1f} dB)")
        ax.plot([f_n], [10 * np.log10(P[np.argmin(abs(f - f_n))] / bg)],
                "o", mfc="white", mec=K, mew=1.4, ms=8,
                label=f"notched max: {f_n:.1f} Hz ({p_n:.1f} dB)")
        ax.set_ylabel("power above background (dB)")
        legend_above(ax, ncol=2, y=1.03)
        panel(ax, letter)
        ax.set_xlim(20, 90)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 3)

    a2.set_xlabel("frequency (Hz)   —   grey bands = multiples of the 6 Hz drive (notched out)")
    save(fig, "figure3_harmonic_comb")


# =============================================================================
# FIG 4 — MAIN RESULT. Pre-registered PING test, fresh seeds 200-204.
# Numbers from results_*.log (ping_scaling_test, 8 tau x 5 seeds x 2 conditions).
# =============================================================================
def figure4():
    print("figure 4: MAIN — period vs tau_GABA, pre-registered (from logged run)")
    tau = np.array([2, 3, 4, 6, 8, 12, 16, 24], float)

    base_f = np.array([39.3, 33.8, 40.5, 38.2, 36.6, 32.2, 37.5, 40.6])
    scal_f = np.array([61.8, 55.2, 55.2, 44.1, 40.4, 35.0, 32.8, 31.0])
    scal_p = np.array([0.259, 0.039, 0.015, 0.001, 0.001, 0.001, 0.001, 0.001])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.5))
    fig.subplots_adjust(wspace=0.30, top=0.78)

    # --- left: frequency. This is the view that misled two earlier criteria. ---
    a1.plot(tau, base_f, "s--", color=G, lw=1.2, ms=5,
            label="80 + 20 neurons")
    a1.plot(tau, scal_f, "o-", color=K, lw=1.5, ms=5.5,
            label="320 + 80 neurons")
    # A log x-axis compresses the high-tau end, where four of the eight grid points
    # sit, and eight labels overlap there. Label a readable subset (the endpoints
    # and the powers of two); all eight points are still drawn and still fitted.
    LABELLED = {2, 4, 8, 16, 24}
    a1.set_xscale("log")
    a1.set_xticks(tau)
    a1.set_xticklabels([f"{t:.0f}" if t in LABELLED else "" for t in tau])
    # bottom=False hides the minor TICK but not its LABEL, so matplotlib's log
    # formatter still prints the decade minor tick at 20 as "2x10^1" between the
    # 16 and 24 labels -- a third notation, on an axis whose other labels are
    # plain integers. labelbottom=False removes it.
    a1.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    a1.set_xlabel(r"$\tau_{\mathrm{GABA}}$ (ms)")
    a1.set_ylabel("peak frequency (Hz)")
    legend_above(a1)
    panel(a1, "A")
    a1.set_ylim(28, 68)

    # --- right: PERIOD. This is the prediction, and the test. ---
    # Panel B originally carried a legend at LOWER RIGHT, while every other panel
    # in the paper has it above. That is the same "place it wherever it fits"
    # had already written the rule down before breaking it here.
    # The slope, R^2 and p are in the caption, where a reader looks for them.
    # They do not need to be printed twice.
    for f_, lab, c, mk, ls in [
        (base_f, "80 + 20 neurons", G, "s", "--"),
        (scal_f, "320 + 80 neurons", K, "o", "-"),
    ]:
        T = 1000.0 / f_
        reg = stats.linregress(tau, T)
        a2.plot(tau, T, mk, color=c, ms=5.5, mfc=("white" if c == G else c),
                mec=c, mew=1.2)
        xx = np.linspace(0, 26, 50)
        a2.plot(xx, reg.intercept + reg.slope * xx, ls, color=c, lw=1.4)
        if c == K:
            a2.plot([0], [reg.intercept], "o", mfc="white", mec=K, mew=1.6, ms=7,
                    zorder=5, clip_on=False)
            a2.annotate(f"intercept {reg.intercept:.1f} ms\n= fixed loop delay",
                        xy=(0, reg.intercept), xytext=(2.2, 35.5), fontsize=7,
                        ha="left", va="top",
                        arrowprops=dict(arrowstyle="->", lw=0.8, color=K))

    a2.set_xlabel(r"$\tau_{\mathrm{GABA}}$ (ms)")
    a2.set_ylabel("cycle period (ms)")
    panel(a2, "B")
    a2.set_xlim(-0.5, 26)
    a2.set_ylim(14, 36)

    save(fig, "figure4_ping_scaling")

    # numbers for the caption
    for f_, lab in [(base_f, "80+20"), (scal_f, "320+80")]:
        reg = stats.linregress(tau, 1000.0 / f_)
        print(f"    {lab}: period = {reg.intercept:.1f} + {reg.slope:.3f}*tau  "
              f"slope {reg.slope:+.3f}±{reg.stderr:.3f}  R2={reg.rvalue**2:.3f}  "
              f"p={reg.pvalue:.4f}")


# =============================================================================
# FIG 5 — screens cannot rescue a masked rhythm. artifact_demo's E3, recomputed.
#
# Same data, two pipelines. The raw statistic followed by all four screens rejects
# a genuine 40 Hz rhythm: the peak it is handed sits on the 30 Hz comb tooth, and
# every screen is correct about THAT peak. A screen can only reject a peak the
# statistic has already found.
# =============================================================================
def figure5():
    print("figure 5: screens cannot rescue a masked rhythm (recomputing, ~2 min)")
    cases = [("no rhythm",        "poisson", dict(depth_theta=0.0), None),
             ("theta-locked\nσ = 5 ms", "sync", dict(jitter_ms=5),  None),
             ("+ real 40 Hz",     "gt",      dict(gamma_hz=40.0),   40.0),
             ("+ real 55 Hz",     "gt",      dict(gamma_hz=55.0),   55.0)]

    def peaks(gen, kw, notch):
        out = []
        for k in range(6):
            rng = np.random.default_rng(3000 + k)
            st = (A.ground_truth(A.T_S, rng, **kw) if gen == "gt" else
                  A.synchronous_spikes(A.BASE_HZ, rng=rng, **kw) if gen == "sync"
                  else A.poisson_spikes(A.BASE_HZ, rng=rng, **kw))
            # N_SURR_SWEEP (400), not E3's N_SURR (2000). The quantity PLOTTED is
            # the peak frequency, which is read off the observed spectrum and does
            # not depend on the surrogates at all. The surrogates enter only through
            # the verdict, and the verdicts here are not close calls: the p-values
            # are 0.000 and 0.5, not 0.04 and 0.06. Stated so that "the figure used
            # a different N_surr from the table" is a disclosed choice with a reason
            # rather than a discrepancy a reader has to discover.
            out.append(A.test_rhythm(st, A.N_NEURON, A.T_S, band=A.GAMMA,
                                     n_surr=A.N_SURR_SWEEP, method="jitter", seed=k,
                                     drive_hz=A.DRIVE_HZ, notch=notch))
        return (float(np.median([r["fpk"] for r in out])),
                np.mean([r["is_rhythm"] for r in out]) > 0.5)

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    fig.subplots_adjust(top=0.78)
    y = np.arange(len(cases))[::-1]
    for i, (name, gen, kw, truth) in enumerate(cases):
        fr, rr = peaks(gen, kw, False)
        fn, rn = peaks(gen, kw, True)
        yy = y[i]
        if truth:
            ax.axvline(truth, color=LG, lw=7, zorder=0,
                       label="the rhythm that is actually there" if i == 2 else None)
        ax.plot(fr, yy + 0.17, "v", color=K, ms=9, mfc="white", mew=1.4,
                label="raw statistic + all four screens" if i == 0 else None)
        ax.plot(fn, yy - 0.17, "o", color=K, ms=8,
                label="comb-notched + the same four screens" if i == 0 else None)
        for f_, dy, ok in ((fr, 0.17, rr == bool(truth)),
                           (fn, -0.17, rn == bool(truth))):
            ax.annotate(f"{f_:.1f}" + ("" if ok else "  rejected"), (f_, yy + dy),
                        xytext=(7, -3), textcoords="offset points", fontsize=7.5,
                        style="normal" if ok else "italic")
    ax.axvline(30.0, color=K, lw=0.9, ls=":")
    ax.annotate("30 Hz = 5 × 6 Hz,\nthe drive's fifth harmonic",
                xy=(41.8, 2.55), fontsize=7.5, ha="left", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in cases], fontsize=8)
    ax.set_xlabel("peak frequency the pipeline reports (Hz)")
    ax.set_xlim(25, 64)
    ax.set_ylim(-0.55, len(cases) - 0.2)
    legend_above(ax, ncol=2)
    save(fig, "figure5_screens_cannot_rescue")


if __name__ == "__main__":
    print_version_banner()
    print()
    print("Building figures.\n")
    figure1()
    figure2()
    figure3_inline()
    figure4()
    figure5()
    print("\nDone. figure1..figure5 as .pdf (manuscript) and .png (preview).")
