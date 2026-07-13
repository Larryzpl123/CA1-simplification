#!/usr/bin/env python3
"""
make_figures.py — the four figures.
Larry (Peilin) Zhong

Fig 1  the null of the prominence statistic          (generated live, no network)
Fig 2  synchrony defeats the surrogate null          (from the logged runs)
Fig 3  the harmonic comb, and the notch              (generated live, no network)
Fig 4  MAIN: period vs tau_GABA -- the PING test     (from the pre-registered run)

Figs 1 and 3 are recomputed from scratch every time this script runs, because
they cost nothing and a figure that cannot be regenerated is a claim you cannot
check. Figs 2 and 4 report network runs that take an hour; their numbers are
transcribed from results_*.log and are marked as such below. Both were reproduced
to the last digit on two machines with different Python, numpy and hardware.

Output: figure1..figure4 as .pdf (vector, for the manuscript) and .png (preview).
"""

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

# ONE legend rule, applied everywhere: OUTSIDE the axes, above, horizontal.
# Earlier drafts placed legends "wherever they fit" -- upper left here, lower
# right there -- which is not a style, it is the absence of one, and it collided
# with the data in three of four figures. A rule that cannot collide is better
# than a judgement call made four separate times.
def legend_above(ax, ncol=2, y=1.02):
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, y),
              ncol=ncol, fontsize=7.5, handlelength=1.8, columnspacing=1.4,
              borderaxespad=0)


def panel(ax, letter):
    """Panel letter only. NO axes titles anywhere in this paper.

    An axes title and a legend-above occupy the same strip and collide, and
    stacking them wastes a third of the figure. Journals put the description in
    the caption for exactly this reason. So: letter in the corner, legend above,
    everything else in the caption. One rule, four figures, no judgement calls."""
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


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

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.hist(v, bins=28, color=LG, edgecolor=K, linewidth=0.6)
    ax.axvline(0, color=K, lw=1.2, ls="-")
    ax.annotate("what an unbiased\nstatistic would give", xy=(0, 0),
                xytext=(0.6, 18), fontsize=7.5, ha="left", color=K,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=K))
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.30)          # headroom so nothing sits on the data
    p95 = np.percentile(v, 95)
    ax.axvline(p95, color=K, lw=1.0, ls="--")
    ax.annotate(f"95th pct {p95:.2f} dB", xy=(p95, ymax * 1.02),
                xytext=(p95 + 1.5, ymax * 1.20), fontsize=7.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=K))

    # where v1's reported values fall — labels staggered so they never touch a line
    for val, lab, yf in [(6.9, "v1 theta 6.6–7.5 dB", 0.86),
                         (11.5, "v1 gamma 11.3–11.9 dB", 0.70)]:
        ax.axvline(val, color=K, lw=1.0, ls=":")
        ax.annotate(lab, xy=(val, ymax * yf), xytext=(val - 0.35, ymax * yf),
                    fontsize=7.5, ha="right", va="center")

    ax.set_xlabel("peak prominence (dB) — 200 datasets containing NO rhythm")
    ax.set_ylabel("count")
    ax.set_xlim(-0.5, 13.5)
    save(fig, "figure1_null_of_statistic")
    return v


# =============================================================================
# FIG 2 — synchrony defeats the null; the notch fixes it.
# Numbers from results_*.log (artifact_demo, 21 s, N_surr = 400).
# =============================================================================
def figure2():
    print("figure 2: synchrony defeats the surrogate null (from logged run)")
    jit = np.array([50, 30, 20, 12, 8, 5, 3], float)
    raw_prom = np.array([5.66, 5.84, 5.59, 12.28, 25.86, 32.98, 37.23])
    notch_prom = np.array([5.69, 5.83, 5.66, 6.34, 6.05, 5.92, 5.44])
    raw_fp = np.array([0, 17, 0, 100, 100, 100, 100], float)
    notch_fp = np.array([0, 17, 0, 17, 0, 0, 0], float)
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
    a2.text(len(jit) - 0.45, 9, "α = 5%", fontsize=7.5, ha="right", va="bottom")
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
    a1.set_xscale("log")
    a1.set_xticks(tau); a1.set_xticklabels([f"{t:.0f}" for t in tau])
    a1.set_xlabel(r"$\tau_{\mathrm{GABA}}$ (ms)")
    a1.set_ylabel("peak frequency (Hz)")
    legend_above(a1)
    panel(a1, "A")
    a1.set_ylim(28, 68)

    # --- right: PERIOD. This is the prediction, and the test. ---
    for f_, lab, c, mk, ls in [
        (base_f, "80 + 20 neurons", G, "s", "--"),
        (scal_f, "320 + 80 neurons", K, "o", "-"),
    ]:
        T = 1000.0 / f_
        reg = stats.linregress(tau, T)
        a2.plot(tau, T, mk, color=c, ms=5.5, mfc=("white" if c == G else c),
                mec=c, mew=1.2)
        xx = np.linspace(0, 26, 50)
        a2.plot(xx, reg.intercept + reg.slope * xx, ls, color=c, lw=1.4,
                label=(f"{lab}\n"
                       f"slope {reg.slope:+.3f} ± {reg.stderr:.3f}, "
                       f"$R^2$={reg.rvalue**2:.2f}, p={reg.pvalue:.4f}"))
        if c == K:
            a2.plot([0], [reg.intercept], "o", mfc="white", mec=K, mew=1.6, ms=7,
                    zorder=5, clip_on=False)
            a2.annotate(f"intercept {reg.intercept:.1f} ms\n= fixed loop delay",
                        xy=(0, reg.intercept), xytext=(1.2, 33.0), fontsize=7,
                        ha="left", va="top",
                        arrowprops=dict(arrowstyle="->", lw=0.8, color=K))

    a2.set_xlabel(r"$\tau_{\mathrm{GABA}}$ (ms)")
    a2.set_ylabel("cycle period (ms)")
    panel(a2, "B")
    a2.legend(frameon=False, loc="lower right", fontsize=6.6,
              handlelength=1.6, labelspacing=0.8, borderaxespad=0.3)
    a2.set_xlim(-0.5, 26)
    a2.set_ylim(14, 39)

    save(fig, "figure4_ping_scaling")

    # numbers for the caption
    for f_, lab in [(base_f, "80+20"), (scal_f, "320+80")]:
        reg = stats.linregress(tau, 1000.0 / f_)
        print(f"    {lab}: period = {reg.intercept:.1f} + {reg.slope:.3f}*tau  "
              f"slope {reg.slope:+.3f}±{reg.stderr:.3f}  R2={reg.rvalue**2:.3f}  "
              f"p={reg.pvalue:.4f}")


if __name__ == "__main__":
    # This script was the ONLY one that never printed its versions, because it is
    # not part of run_all.sh's battery. So no log recorded matplotlib, so when the
    # frozen requirements were written its version had to be guessed -- in a
    # repository whose whole subject is not reporting unmeasured numbers.
    # Every script prints what it loaded now. No exceptions.
    print_version_banner()
    print()
    print("Building figures. Figs 1 and 3 are recomputed; 2 and 4 are the logged runs.\n")
    figure1()
    figure2()
    figure3_inline()
    figure4()
    print("\nDone. figure1..figure4 as .pdf (manuscript) and .png (preview).")
