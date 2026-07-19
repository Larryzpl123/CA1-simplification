#!/usr/bin/env python3
"""
make_figures.py — the six figures.
Larry (Peilin) Zhong

Numbered in the order they appear in the manuscript, which is why the function
that draws figure 3 is defined as figure3_inline (the harmonic-comb figure moved
after the masking figure when §3.5 was added):

Fig 1  the null of the prominence statistic
Fig 2  synchrony defeats the surrogate null
Fig 3  screens cannot rescue a rhythm the statistic never found (§3.5)
Fig 4  the harmonic comb, and the notch (§3.6)
Fig 5  MAIN: period vs tau_GABA -- the pre-registered PING test (§3.8)
Fig 6  the switch is the input fluctuation, not the cell count (§3.8)

Figures 1-4 and 6's markers are recomputed or parsed live. Figures 5 and 6 draw
the network sweeps from ping_scaling_test.py and job_b_noise_matched.py, which
take ~25 and ~35 minutes, so they are not re-run here: they PARSE the newest run
logs in ../results/ for the per-tau peaks and for the fit each test reported, and
draw that fit rather than computing one of their own. Nothing in this file carries
a copy of a result.

Writes to ../figures/. That is the only place the figures live.
Output: figure1..figure6 as .pdf (vector, for the manuscript) and .png (preview).
"""

import os
import numpy as np
import glob
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import sys
sys.path.insert(0, ".")
from spectral_null import (pop_rate, psd, prominence, prominence_notched, DT_S,
                           print_version_banner)
import artifact_demo as A


# ---------------- figure 4 reads a run log; it does not carry its own copy ----
# A figure that hard-codes numbers is a second, unverified record of the result,
# and it does not recompute itself when the code underneath it changes. That is
# how Figure 2 went stale in this archive. These two helpers are why Figure 4's
# "(from logged run)" is now a true statement rather than a label.
def newest_log(pattern="../results/*.log"):
    logs = [p for p in glob.glob(pattern) if "PING SCALING TEST" in open(p, encoding="utf-8").read()]
    assert logs, f"no run log with a PING section matches {pattern}"
    return max(logs, key=os.path.getmtime)

def ping_from_log(path):
    """Per-tau median peaks AND the fit the test reported, for both conditions."""
    t = open(path, encoding="utf-8").read()
    sec = t[t.index("PING SCALING TEST"):]
    out = {}
    for name in ("LIF baseline", "Scaled 4x"):
        blk = sec[sec.index("\n" + name + "\n"):]
        blk = blk[:blk.index("PRE-REGISTERED CRITERION")]
        rows = re.findall(r'^\s*(\d+)ms\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+'
                          r'[\d.-]+\s+[\d.-]+\s+([\d.]+)\s+\[', blk, re.M)
        fit = re.search(r'fitted:\s+period = ([\d.+-]+) \+ ([\d.+-]+) \* tau_GABA', blk)
        sta = re.search(r'slope = ([\d.+-]+) \+/- ([\d.]+)\s+R\^2 = ([\d.]+)\s+p = ([\d.]+)', blk)
        assert rows and fit and sta, f"could not parse the {name} block of {path}"
        out[name] = dict(tau=np.array([float(a) for a, _ in rows]),
                         f=np.array([float(b) for _, b in rows]),
                         a=float(fit.group(1)), k=float(fit.group(2)),
                         stderr=sta.group(2), r2=sta.group(3), p=sta.group(4))
    return out

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
    save(fig, "figure4_harmonic_comb")


# =============================================================================
# FIG 4 — MAIN RESULT. Pre-registered PING test, fresh seeds 200-204.
# Numbers from results_*.log (ping_scaling_test, 8 tau x 5 seeds x 2 conditions).
# =============================================================================
def figure4():
    print("figure 4: MAIN — period vs tau_GABA, pre-registered (parsed from a run log)")

    # This used to read:
    #     base_f = np.array([39.3, 33.8, 40.5, ...])
    #     scal_f = np.array([61.8, 55.2, 55.2, ...])
    # hand-copied out of a log, under a print that claimed "(from logged run)".
    # The label was false: nothing connected those arrays to any log, so
    # re-running ping_scaling_test.py would have changed the numbers in the paper
    # and left the MAIN figure drawing the old ones, silently. That is not
    # hypothetical. It already happened once in this archive, to Figure 2, and
    # the v2.2 release notes record it: "Figure 2 now COMPUTES E4 instead of
    # transcribing (it had gone stale)." Figure 4 was left transcribing.
    #
    # It also used to re-run its own stats.linregress on those copied values. The
    # log prints the medians rounded to one decimal, so that second regression
    # was fitted to rounded data and disagreed with the test's own fit in the
    # third decimal: +0.761 in the figure against +0.760 in the paper. Nothing
    # displayed it, so nothing caught it. A figure has no business re-deriving a
    # result: it plots the fit the test reported, or it is a second experiment
    # with no criterion.
    log = newest_log()
    d = ping_from_log(log)
    print(f"    source: {os.path.basename(log)}")
    tau    = d["Scaled 4x"]["tau"]
    base_f = d["LIF baseline"]["f"]
    scal_f = d["Scaled 4x"]["f"]
    assert np.array_equal(tau, d["LIF baseline"]["tau"]), \
        "the two conditions were not swept on the same tau_GABA grid"

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
    for f_, key, c, mk, ls in [
        (base_f, "LIF baseline", G, "s", "--"),
        (scal_f, "Scaled 4x",    K, "o", "-"),
    ]:
        T = 1000.0 / f_
        a_fit, k_fit = d[key]["a"], d[key]["k"]     # the fit the TEST reported
        a2.plot(tau, T, mk, color=c, ms=5.5, mfc=("white" if c == G else c),
                mec=c, mew=1.2)
        # Draw the fit ONLY across the range that was swept. This used to run
        # np.linspace(0, 26), which continued the line to tau_GABA = 0 and put a
        # marker and an "= fixed loop delay" label on the intercept there. At
        # tau_GABA = 0 there is no PING (Borgers, pers. comm., 2026-07-16), so
        # that was a picture of the model in a regime the model does not
        # describe, and the label read a mechanism out of the extrapolation.
        # A figure should not draw a claim the text has withdrawn.
        xx = np.linspace(tau.min(), tau.max(), 50)
        a2.plot(xx, a_fit + k_fit * xx, ls, color=c, lw=1.4)

    a2.set_xlabel(r"$\tau_{\mathrm{GABA}}$ (ms)")
    a2.set_ylabel("cycle period (ms)")
    panel(a2, "B")
    # Start the axis at the smallest tau actually swept, not at 0. An axis that
    # shows tau_GABA = 0 invites the eye to run the line down to it, which is
    # the extrapolation this figure no longer makes.
    a2.set_xlim(1.0, 25.5)
    a2.set_ylim(14, 36)

    save(fig, "figure5_ping_scaling")

    # The numbers for the caption, echoed from the log so they can be checked
    # against the figure without opening it.
    #
    # This block used to run stats.linregress here and print ITS result under the
    # heading "numbers for the caption". Those were a regression on the medians
    # AS ROUNDED BY THE LOG, so they disagreed with the test's own fit in the
    # third decimal -- it printed slope +0.761, R2 = 0.883, p = 0.7507 where
    # ping_scaling_test.py reports +0.760, 0.884 and 0.7482. A script that labels
    # a number "for the caption" and hands you a different number from the one
    # the paper must quote is not a convenience. It is a transcription error
    # waiting to be made, by the person who trusts it most.
    for key, lab in [("LIF baseline", "80+20"), ("Scaled 4x", "320+80")]:
        v = d[key]
        print(f"    {lab}: period = {v['a']:.1f} + {v['k']:.3f}*tau   "
              f"slope {v['k']:+.3f} +/- {v['stderr']}   R2 = {v['r2']}   "
              f"p = {v['p']}      [echoed from {os.path.basename(log)}]")


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
    save(fig, "figure3_screens_cannot_rescue")


# =============================================================================
# FIG 6 — the switch is the input fluctuation, not the cell count.
#
# Three period-vs-tau_GABA fits, all PARSED from the Job B run log, never
# transcribed and never re-fitted here (same rule as Figure 4): 320+80 control,
# 80+20 at base connection probabilities (fails), and 80+20 with each pathway's
# coefficient of variation matched to the 320+80 network's (passes). The two
# 80+20 rows have the same cells; only the input fluctuations differ.
#
# The 80+20 base row is not in the Job B log (it was the v4 ping_scaling run), so
# it is parsed from the newest log that contains a "LIF baseline" PING block.
# =============================================================================
def _newest_with(pattern, needle):
    """Newest matching log that contains `needle`."""
    hits = [p for p in glob.glob(pattern)
            if needle in open(p, encoding="utf-8").read()]
    assert hits, f"no log matching {pattern} contains {needle!r}"
    return max(hits, key=os.path.getmtime)


def _ping_block(log_text, marker):
    """Per-tau median peaks and the reported fit for the block whose header
    contains `marker`. `marker` must be unique in the log (e.g. 'control, base p',
    'noise-matched p', or the standalone 'LIF baseline' condition heading), not a
    bare substring like '80+20' that also appears in surrounding prose."""
    assert log_text.count(marker) >= 1, f"marker {marker!r} not in log"
    blk = log_text[log_text.index(marker):]
    blk = blk[:blk.index("PRE-REGISTERED CRITERION")]
    rows = re.findall(r'^\s*(\d+)ms\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+'
                      r'(?:[\d.-]+\s+[\d.-]+\s+)?([\d.]+)\s+\[', blk, re.M)
    fit = re.search(r'period = ([\d.+-]+) \+ ([\d.+-]+) \* tau_GABA', blk)
    assert rows and fit, f"could not parse the {name!r} block"
    return (np.array([float(a) for a, _ in rows], float),
            np.array([float(b) for _, b in rows], float),
            float(fit.group(1)), float(fit.group(2)))


def figure6():
    print("figure 6: the switch is the fluctuation, not the count (from logged runs)")
    job = _newest_with("../results/jobB_*.log", "80+20")
    base = _newest_with("../results/*.log", "LIF baseline")   # the ping_scaling run
    jt = open(job, encoding="utf-8").read()
    bt = open(base, encoding="utf-8").read()
    print(f"    control + matched: {os.path.basename(job)}")
    print(f"    base 80+20:        {os.path.basename(base)}")

    tau_c, f_c, a_c, k_c = _ping_block(jt, "control, base p")
    tau_m, f_m, a_m, k_m = _ping_block(jt, "noise-matched p")
    tau_b, f_b, a_b, k_b = _ping_block(bt, "LIF baseline")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fig.subplots_adjust(top=0.78)
    series = [
        (tau_c, f_c, a_c, k_c, K, "o", "-",  "320 + 80  (control)"),
        (tau_b, f_b, a_b, k_b, G, "s", "--", "80 + 20  (base p, fails)"),
        (tau_m, f_m, a_m, k_m, K, "^", ":",  "80 + 20  (fluctuation matched)"),
    ]
    for tau, f_, a_fit, k_fit, c, mk, ls, lab in series:
        T = 1000.0 / f_
        ax.plot(tau, T, mk, color=c, ms=5.5,
                mfc=("white" if mk in ("s", "^") else c), mec=c, mew=1.2,
                label=lab)
        xx = np.linspace(tau.min(), tau.max(), 50)
        ax.plot(xx, a_fit + k_fit * xx, ls, color=c, lw=1.4)

    ax.set_xscale("log")
    ax.set_xticks([2, 4, 8, 16, 24])
    ax.set_xticklabels(["2", "4", "8", "16", "24"])
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.set_xlabel(r"$\tau_{\mathrm{GABA}}$ (ms)")
    ax.set_ylabel("cycle period (ms)")
    ax.set_xlim(1.9, 25.5)
    legend_above(ax, ncol=1, y=1.02)
    save(fig, "figure6_fluctuation_not_count")

    # numbers for the caption, echoed from the logs.
    for lab, key, txt in [("320+80 control", "control, base p", jt),
                          ("80+20 matched",  "noise-matched p", jt),
                          ("80+20 base",      "LIF baseline",    bt)]:
        blk = txt[txt.index(key):]
        m = re.search(r"slope = ([\d.+-]+) \+/- ([\d.]+)\s+"
                      r"R\^2 = ([\d.]+)\s+p = ([\d.]+)", blk)
        if m:
            print(f"    {lab:15} slope {m.group(1)} +/- {m.group(2)}   "
                  f"R2 = {m.group(3)}   p = {m.group(4)}")


if __name__ == "__main__":
    print_version_banner()
    print()
    print("Building figures.\n")
    figure1()
    figure2()
    figure3_inline()
    figure4()
    figure5()
    figure6()
    print("\nDone. figure1..figure6 as .pdf (manuscript) and .png (preview).")
