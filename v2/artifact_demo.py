#!/usr/bin/env python3
"""
artifact_demo.py — the generality experiment.
Larry (Peilin) Zhong

WHY THIS FILE IS THE CORE OF THE PAPER
--------------------------------------
The obvious objection to this work is: "these are bugs in YOUR model, not a
finding." That objection is fatal if the artifacts are specific to one
implementation. So this file removes the model entirely.

There is NO network here. No neurons, no synapses, no Brian2. Just synthetic
spike trains drawn from an inhomogeneous Poisson process, in which the presence
or absence of a gamma rhythm is known BY CONSTRUCTION because we put it there
(or did not). If the artifacts appear here, they are properties of

    (i)  the peak-prominence statistic itself, and
    (ii) theta-locked spiking itself,

and every reduced-network study that uses this statistic on theta-driven spiking
inherits them, whoever wrote the code.

THE EXPERIMENTS
---------------
E1  the null of the statistic, and how it moves with the frequency resolution
E2  theta harmonics masquerading as gamma: RATE modulation alone suffices
E3  screens cannot rescue a rhythm the statistic never found
E4  synchrony, and how it defeats the surrogate null completely
E5  the peak is a harmonic comb, not a band-edge artifact
E6  the comb does not only create false positives -- it masks real rhythms
E7  validation of the repair on ground truth

E7 aborts the run if the repair fails. E3 aborts if it and E6 disagree.

Run:  python3 artifact_demo.py
"""

import numpy as np
from spectral_null import (test_rhythm, pop_rate, psd, prominence,
                           prominence_notched, harmonic_of, DT_S)

T_S      = 21.0         # seconds. Set by the resolution requirement: df must
                        # satisfy 3*df/f_drive <= 0.10 (spectral_null.required_df).
                        # At 2.5 s the harmonic screen blocks 20% of the band and
                        # destroys a ground-truth 40 Hz rhythm. At 21 s, 6%.
N_NEURON = 80
BASE_HZ  = 6.0          # mean firing rate
DRIVE_HZ = 6.0          # the imposed theta, as in the CA1 model
GAMMA    = (30.0, 80.0)
N_SURR   = 2000
N_REPS   = 200


def poisson_spikes(rate_hz, depth_theta=0.0, depth_gamma=0.0, gamma_hz=40.0,
                   sharpness=1.0, rng=None, T=T_S, n=N_NEURON, dt=DT_S):
    """Inhomogeneous Poisson population: each neuron samples INDEPENDENTLY from a
    common smooth rate lambda(t).

    IMPORTANT: rate modulation alone IS sufficient, if the drive is sharp.
    Measured un-notched (E2): at depth 0.9 and sharpness 10 these trains produce
    7.55 dB at exactly 5 x 6 Hz, significant against the surrogate null in 83% of
    datasets, with gamma content of zero and no synchrony whatsoever. Sharpening
    lambda(t) pushes power into higher harmonics faster than shot noise buries
    them. The harmonic screen catches these; the surrogate null does not.

    Synchrony (synchronous_spikes, below) is the more dangerous route: it produces
    12-37 dB rather than 7.6, so it defeats not only the null but any prominence
    threshold substituted for one."""
    rng = rng or np.random.default_rng(0)
    t = np.arange(0, T, dt)
    lam = np.full_like(t, float(rate_hz))
    if depth_theta:
        th = (0.5 + 0.5 * np.sin(2 * np.pi * DRIVE_HZ * t)) ** sharpness
        lam = lam * (1 - depth_theta + 2 * depth_theta * th)
    if depth_gamma:
        lam = lam * (1 + depth_gamma * np.sin(2 * np.pi * gamma_hz * t))
    lam = np.clip(lam, 0, None)
    counts = rng.poisson(lam * n * dt)
    idx = np.nonzero(counts)[0]
    return np.repeat(t[idx], counts[idx])


def synchronous_spikes(rate_hz, jitter_ms, depth_gamma=0.0, gamma_hz=40.0,
                       rng=None, T=T_S, n=N_NEURON, dt=DT_S):
    """The mechanism that ACTUALLY produces the artifact: SYNCHRONY, not rate.

    Real (and simulated) theta-driven neurons do not sample independently. They
    have thresholds and refractory periods and share a common oscillating drive,
    so they fire in TIGHT, PHASE-LOCKED VOLLEYS. The population rate is then a
    near-deterministic pulse train at 6 Hz, whose Fourier series carries strong
    harmonics far into the gamma band -- because a narrow pulse of width sigma
    has spectral content out to ~1/sigma.

    Here each spike is placed on a theta cycle with Gaussian jitter of sigma =
    jitter_ms. jitter_ms is therefore an explicit, dial-able measure of how
    tightly the population is locked. depth_gamma=0 means NO gamma exists."""
    rng = rng or np.random.default_rng(0)
    n_cycles = int(T * DRIVE_HZ)
    spikes_per_cycle = max(int(round(rate_hz * n / DRIVE_HZ)), 1)
    st = []
    for c in range(n_cycles):
        centre = (c + 0.25) / DRIVE_HZ                 # locked to theta phase
        s = centre + rng.normal(0, jitter_ms / 1000.0, spikes_per_cycle)
        st.append(s)
    st = np.concatenate(st)
    if depth_gamma:                                    # optional REAL gamma
        st = st + (depth_gamma / (2 * np.pi * gamma_hz)) * np.sin(
            2 * np.pi * gamma_hz * st)
    st = st[(st >= 0) & (st < T)]
    return np.sort(st)


def volleys(f_hz, jitter_ms, n_per_cycle, T, rng, phase=0.25):
    n_cyc = int(T * f_hz)
    st = [((c + phase) / f_hz) + rng.normal(0, jitter_ms / 1000.0, n_per_cycle)
          for c in range(n_cyc)]
    st = np.concatenate(st)
    return st[(st >= 0) & (st < T)]


def ground_truth(T, rng, theta_jit=5.0, gamma_hz=None, gamma_jit=3.0,
                 gamma_frac=0.5, n_total=N_NEURON * BASE_HZ):
    """UNION OF TWO INDEPENDENT SOURCES: a 6 Hz theta volley train, plus optionally
    a genuine gamma volley train. This is the correct positive control.

    Gamma must be injected as an INDEPENDENT source. Phase-modulating spike times
    with a sinusoid does not add a rhythm: it multiplies the 6 Hz comb by the
    injected frequency, and the intermodulation product is what appears. Measured
    (v1_diagnosis.py, M1): a nominal 40 Hz injection peaks at 36.0 Hz and a nominal
    55 Hz at 48.0 Hz. A positive control that does not contain the rhythm it claims
    to contain is worthless."""
    n_theta = int(round(n_total * ((1 - gamma_frac) if gamma_hz else 1.0) / DRIVE_HZ))
    st = [volleys(DRIVE_HZ, theta_jit, max(n_theta, 1), T, rng)]
    if gamma_hz:
        n_g = max(int(round(n_total * gamma_frac / gamma_hz)), 1)
        st.append(volleys(gamma_hz, gamma_jit, n_g, T, rng, phase=0.5))
    return np.sort(np.concatenate(st))


def prom_only(st, rng_unused=None):
    r = pop_rate(st, N_NEURON, T_S)
    f, P = psd(r, 1.0 / DT_S)
    return prominence(f, P, GAMMA)


# ===========================================================================
def e1_null_of_statistic():
    print("=" * 84)
    print("E1  NULL OF THE PROMINENCE STATISTIC  (constant-rate Poisson; NO rhythm)")
    print("=" * 84)
    vals = []
    for k in range(N_REPS):
        st = poisson_spikes(BASE_HZ, rng=np.random.default_rng(1000 + k))
        vals.append(prom_only(st)[0])
    v = np.array(vals)
    print(f"  {N_REPS} independent datasets containing NO rhythm whatsoever.")
    print(f"  An unbiased statistic would return ~0 dB. It returns:")
    print(f"      mean   {v.mean():5.2f} dB")
    print(f"      sd     {v.std():5.2f} dB")
    print(f"      95th   {np.percentile(v, 95):5.2f} dB")
    print(f"      max    {v.max():5.2f} dB")
    print()
    print(f"  => Any 'gamma prominence' below ~{np.percentile(v, 95):.1f} dB is not evidence.")
    print(f"     v1 of this preprint reported theta prominence of 6.6-7.5 dB and")
    print(f"     gamma prominence of 11.3-11.9 dB against an implicit null of 0 dB.")
    for ref in (5.0, 6.9, 7.5, 11.5):
        pct = 100.0 * np.mean(v >= ref)
        print(f"        a reported {ref:4.1f} dB is exceeded by {pct:5.1f}% of NO-rhythm data")
    print()

    # THE NULL IS NOT A CONSTANT. It depends on the analysis, because the statistic
    # is a MAXIMUM over bins and finer resolution means more bins to maximise over.
    # The manuscript quoted a value at a coarse resolution that no script produced.
    # It is measured here, at both resolutions, so the claim has a source.
    print("  The null MOVES WITH THE ANALYSIS. Same data, same statistic, only")
    print("  the Welch segment length changes:\n")
    print(f"  {'nperseg':>9} {'df (Hz)':>9} {'bins in 30-80':>14} "
          f"{'null mean':>10} {'null 95th':>10}")
    print("  " + "-" * 58)
    for nps in (1024, 2048, 4096, 8192):
        vv = []
        for k in range(60):
            st = poisson_spikes(BASE_HZ, rng=np.random.default_rng(1000 + k))
            r = pop_rate(st, N_NEURON, T_S)
            fq, P = psd(r, 1.0 / DT_S, nperseg=nps)
            vv.append(prominence(fq, P, GAMMA)[0])
        vv = np.array(vv)
        df = (1.0 / DT_S) / nps
        nb = int((GAMMA[1] - GAMMA[0]) / df)
        print(f"  {nps:>9} {df:>9.3f} {nb:>14} {vv.mean():>9.2f} dB "
              f"{np.percentile(vv, 95):>8.2f} dB")
    print()
    print("  A prominence value is UNINTERPRETABLE unless the sampling rate,")
    print("  segment length and fit range are reported alongside it.")
    print()
    return v


def e2_theta_harmonics(null_v):
    print("=" * 84)
    print("E2  THETA HARMONICS MASQUERADING AS GAMMA  (6 Hz drive; gamma depth = 0)")
    print("=" * 84)
    print("  Spike trains contain a 6 Hz modulation and NOTHING ELSE.")
    print("  Gamma-band content by construction: zero.\n")
    print(f"  {'depth':>6} {'sharp':>6} {'prom dB':>8} {'null p':>8} {'f_pk':>7} "
          f"{'k x 6Hz':>9} {'null says':>11}  {'3 screens say'}")
    print("  " + "-" * 78)
    rows = []
    for depth, sharp in [(0.0, 1), (0.3, 1), (0.6, 1), (0.9, 1),
                         (0.9, 3), (0.9, 6), (0.9, 10)]:
        ps, pr, fp, sig, rhy = [], [], [], [], []
        for k in range(6):
            st = poisson_spikes(BASE_HZ, depth_theta=depth, sharpness=sharp,
                                depth_gamma=0.0,
                                rng=np.random.default_rng(2000 + k))
            # notch=False: this experiment measures the UNREPAIRED statistic.
            # test_rhythm notches by default whenever drive_hz is given.
            r = test_rhythm(st, N_NEURON, T_S, band=GAMMA, n_surr=N_SURR,
                            method="jitter", seed=k, drive_hz=DRIVE_HZ,
                            notch=False)
            ps.append(r["p"]); pr.append(r["obs_db"]); fp.append(r["fpk"])
            sig.append(r["significant"]); rhy.append(r["is_rhythm"])
        f_med = float(np.median(fp))
        kk = int(round(f_med / DRIVE_HZ))
        null_call = f"{100*np.mean(sig):3.0f}% SIG" if np.mean(sig) > 0 else "  0% sig"
        scr_call = f"{100*np.mean(rhy):3.0f}% rhythm"
        print(f"  {depth:>6.1f} {sharp:>6} {np.mean(pr):>8.2f} {np.median(ps):>8.3f} "
              f"{f_med:>7.1f} {kk:>4}x6={kk*6:<4} {null_call:>11}  {scr_call}")
        rows.append((depth, sharp, np.mean(pr), np.median(ps),
                     np.mean(sig), np.mean(rhy)))
    print()
    worst = max(rows, key=lambda r: r[4])
    print(f"  => At theta depth {worst[0]}, sharpness {worst[1]}, spike trains with ZERO gamma")
    print(f"     produce {worst[2]:.1f} dB of 'gamma prominence' and the SURROGATE NULL")
    print(f"     calls them significant in {100*worst[4]:.0f}% of datasets.")
    print(f"     The harmonic screen rejects {100*(1-worst[5]):.0f}% of them.")
    print()
    sharp_rows = [r for r in rows if r[4] > 0.5]
    if sharp_rows:
        print("  RATE MODULATION ALONE IS ENOUGH, if the waveform is sharp enough.")
        print("  These populations")
        print("  sample INDEPENDENTLY from a common rate. There is no synchrony here at")
        print("  all, and the surrogate null still calls the harmonics significant.")
        print("  Sharpening the drive concentrates its Fourier series into higher")
        print("  harmonics, and those land in the gamma band. What the surrogate cannot")
        print("  do is tell a harmonic from a rhythm, and no amount of independence in")
        print("  the spiking rescues it.")
        print()
    print("  THIS IS THE POINT: a surrogate null is NECESSARY AND NOT SUFFICIENT.")
    print("  It asks 'is there non-random fast structure?'. Theta harmonics ARE")
    print("  non-random fast structure. They are real. They are not a rhythm.")
    print()
    return rows


def e3_screens_cannot_rescue_a_masked_rhythm():
    """Screens can only reject a peak the statistic has already found.

    Both statistics are run on the same data:

        raw + screens      catches the artifact, and REJECTS the real rhythm
        notched + screens  catches the artifact, and FINDS the real rhythm

    The raw maximum lands on the 30 Hz comb tooth whether a rhythm exists or not
    (E6), so every screen is correct about the peak it is handed and the rhythm is
    rejected anyway. Screening a maximum that is in the wrong place cannot move
    it."""
    print("=" * 84)
    print("E3  SCREENS CANNOT RESCUE A RHYTHM THE STATISTIC NEVER FOUND")
    print("=" * 84)
    cases = [
        ("no rhythm at all",         "poisson", dict(depth_theta=0.0), False, None),
        ("theta-locked, sigma=5 ms", "sync",    dict(jitter_ms=5),     False, None),
        ("GROUND-TRUTH 40 Hz gamma", "gt",      dict(gamma_hz=40.0),   True,  40.0),
        ("GROUND-TRUTH 55 Hz gamma", "gt",      dict(gamma_hz=55.0),   True,  55.0),
    ]

    def run(gen, kw, notch):
        rs = []
        for k in range(6):
            rng = np.random.default_rng(3000 + k)
            if gen == "gt":
                st = ground_truth(T_S, rng, **kw)
            elif gen == "sync":
                st = synchronous_spikes(BASE_HZ, rng=rng, **kw)
            else:
                st = poisson_spikes(BASE_HZ, rng=rng, **kw)
            rs.append(test_rhythm(st, N_NEURON, T_S, band=GAMMA, n_surr=N_SURR,
                                  method="jitter", seed=k, drive_hz=DRIVE_HZ,
                                  notch=notch))
        return rs

    print(f"  {'case':<26}{'truth':>7} | {'RAW: peak':>10}{'verdict':>11} "
          f"| {'NOTCHED: peak':>14}{'verdict':>11}")
    print("  " + "-" * 80)
    raw_wrong, ntc_wrong = 0, 0
    for name, gen, kw, truth, f_true in cases:
        out = {}
        for tag, notch in (("raw", False), ("ntc", True)):
            rs = run(gen, kw, notch)
            rhy = np.mean([r["is_rhythm"] for r in rs]) > 0.5
            f_ = float(np.median([r["fpk"] for r in rs]))
            # "correct" means: called a rhythm iff one exists AND, if one exists,
            # put it at the right frequency. Finding a real 40 Hz rhythm at 30 Hz
            # is not a success with a rounding error; it is a different claim.
            ok = (rhy == truth) and (not truth or abs(f_ - f_true) < 1.0)
            out[tag] = (f_, rhy, ok)
        (fr, rr, okr), (fn, rn, okn) = out["raw"], out["ntc"]
        raw_wrong += (not okr)
        ntc_wrong += (not okn)
        v = lambda r, o: ("RHYTHM" if r else "rejected") + ("" if o else " X")
        print(f"  {name:<26}{str(truth):>7} | {fr:>8.1f} Hz{v(rr, okr):>11} "
              f"| {fn:>12.1f} Hz{v(rn, okn):>11}")

    print()
    print(f"  raw statistic + four screens : WRONG on {raw_wrong} of 4 cases")
    print(f"  notched statistic + screens  : WRONG on {ntc_wrong} of 4 cases")
    print()
    if raw_wrong == 0:
        print("  *** The raw pipeline got everything right. That contradicts E6, where")
        print("  *** the raw maximum lands on the 30 Hz tooth whether a rhythm exists")
        print("  *** or not. One of the two experiments is broken. Do not report either.")
        raise SystemExit("E3 and E6 disagree -- do not report these results")
    if ntc_wrong:
        print("  *** The REPAIRED pipeline got a case wrong. The repair is not")
        print("  *** validated and nothing downstream may be reported.")
        raise SystemExit(f"E3: notched pipeline wrong on {ntc_wrong} of 4 cases")

    print("  The raw pipeline REJECTS the genuine 40 and 55 Hz rhythms. Every screen")
    print("  fires: significant, at a band edge, a harmonic of the drive. Each of")
    print("  those verdicts is CORRECT about the peak it was handed -- the peak really")
    print("  is at 30.0 Hz, and 30.0 Hz really is 5 x 6 Hz. The screens are not")
    print("  malfunctioning. They are answering a question about the wrong peak.")
    print()
    print("  A SCREEN CAN ONLY REJECT A PEAK THE STATISTIC HAS ALREADY FOUND. When the")
    print("  statistic never finds the rhythm, no number of screens can recover it,")
    print("  and adding more screens makes the pipeline more confident, not more")
    print("  correct. The comb has to come out of the SPECTRUM, before the maximum is")
    print("  taken. That is not a screen. That is a different statistic.")
    print()


# ===========================================================================
# E4-E7: the four results tables in manuscript sections 3.3, 3.4, 3.5 and 3.6.
# Every generator, statistic and screen below is the one already used above.
# ===========================================================================

JITTER_SWEEP = [50, 30, 20, 12, 8, 5, 3]      # ms, sigma of theta phase-locking
N_REP_SWEEP  = 6                              # so a single failure reads as 17%
N_SURR_SWEEP = 400


def e4_synchrony_defeats_the_null():
    """Manuscript section 3.3, and Figure 2.

    Tighten the phase-locking and nothing else. Gamma content is zero in every
    row, by construction. The raw statistic climbs to 37 dB and the surrogate
    null calls it significant in 100% of datasets. The comb-notched statistic,
    scored against a null built with the identical statistic, does not move."""
    print("=" * 84)
    print("E4  SYNCHRONY, NOT RATE, DEFEATS THE SURROGATE NULL   (gamma = 0 in every row)")
    print("=" * 84)
    print(f"  {N_REP_SWEEP} datasets per row, N_surr = {N_SURR_SWEEP}, T = {T_S:.0f} s.\n")
    print(f"  {'sigma':>6} | {'raw dB':>7} {'raw p':>7} {'raw pk':>7} {'raw sig':>8} "
          f"| {'ntch dB':>8} {'ntch p':>7} {'ntch sig':>9}")
    print("  " + "-" * 78)
    for jit in JITTER_SWEEP:
        raw, ntc = [], []
        for k in range(N_REP_SWEEP):
            st = synchronous_spikes(BASE_HZ, jitter_ms=jit, depth_gamma=0.0,
                                    rng=np.random.default_rng(4000 + k))
            raw.append(test_rhythm(st, N_NEURON, T_S, band=GAMMA,
                                   n_surr=N_SURR_SWEEP, method="jitter", seed=k,
                                   drive_hz=DRIVE_HZ, notch=False))
            ntc.append(test_rhythm(st, N_NEURON, T_S, band=GAMMA,
                                   n_surr=N_SURR_SWEEP, method="jitter", seed=k,
                                   drive_hz=DRIVE_HZ, notch=True))
        f = lambda rs, key: np.mean([r[key] for r in rs])
        m = lambda rs, key: np.median([r[key] for r in rs])
        print(f"  {jit:>4} ms | {f(raw,'obs_db'):>7.2f} {m(raw,'p'):>7.3f} "
              f"{m(raw,'fpk'):>6.1f}  {100*f(raw,'significant'):>6.0f}% "
              f"| {f(ntc,'obs_db'):>8.2f} {m(ntc,'p'):>7.3f} "
              f"{100*f(ntc,'significant'):>7.0f}%")
    print()
    print("  A null distribution is NECESSARY AND NOT SUFFICIENT. The surrogate")
    print("  asks whether there is non-random fast temporal structure. There is.")
    print("  It is the harmonics of the drive. It is entirely real. It is not gamma.")
    print()


def e5_peak_is_a_comb_not_a_band_edge():
    """Manuscript section 3.4.

    v1 attributed its 30.0 Hz peaks to a band-edge fitting artifact. A fitting
    artifact can be removed by changing the fit; a harmonic cannot. Move the
    lower edge of the analysis band and watch where the maximum goes: it does
    not follow the edge, it jumps to the next multiple of the drive."""
    print("=" * 84)
    print("E5  THE PEAK IS A HARMONIC COMB, NOT A BAND-EDGE ARTIFACT")
    print("=" * 84)
    st = synchronous_spikes(BASE_HZ, jitter_ms=5, depth_gamma=0.0,
                            rng=np.random.default_rng(5000))
    r = pop_rate(st, N_NEURON, T_S)
    fr, P = psd(r, 1.0 / DT_S)
    print("  ONE tightly-locked, gamma-free train (sigma = 5 ms). Only the band moves.\n")
    print(f"  {'analysis band':>15} | {'peak':>7} {'prominence':>11} | interpretation")
    print("  " + "-" * 66)
    for lo in (30.0, 32.0, 38.0, 44.0, 50.0):
        db, fpk, _ = prominence(fr, P, (lo, 80.0))
        k = int(round(fpk / DRIVE_HZ))
        near = abs(fpk - k * DRIVE_HZ) < 1.0
        tag = f"= {k} x {DRIVE_HZ:.0f} Hz" if near else "NOT a drive harmonic"
        print(f"  {lo:>6.0f}-80 Hz    | {fpk:>6.1f} {db:>10.2f} dB | {tag}")
    print()
    print("  The peak does not follow the band edge. Wherever the band is placed,")
    print("  the maximum lands on a tooth of the comb.")
    print()


GT_CASES = [(None, "no gamma"), (40.0, "real 40 Hz"),
            (55.0, "real 55 Hz"), (68.0, "real 68 Hz")]


def e6_the_comb_masks_real_rhythms():
    """Manuscript section 3.5, and Figure 3. The most important experiment here.

    A screen can only reject a peak the statistic has already found. If the
    statistic never finds the rhythm, no amount of screening can recover it.
    Under tight theta locking the raw maximum reports 30.0 Hz whether a genuine
    40, 55 or 68 Hz rhythm is present or not: its output is very nearly
    uncorrelated with the gamma content it purports to measure."""
    print("=" * 84)
    print("E6  THE COMB DOES NOT ONLY CREATE FALSE POSITIVES -- IT MASKS REAL RHYTHMS")
    print("=" * 84)
    print("  Ground truth = a tight (sigma = 5 ms) 6 Hz volley train, PLUS, in three")
    print("  of four cases, a genuine independent gamma volley train.\n")
    print(f"  {'case':>12} | {'raw dB':>7} {'raw peak':>9} | {'ntch dB':>8} {'ntch peak':>10} | found?")
    print("  " + "-" * 70)
    for gh, label in GT_CASES:
        rng = np.random.default_rng(6000)
        st = ground_truth(T_S, rng, theta_jit=5.0, gamma_hz=gh)
        r = pop_rate(st, N_NEURON, T_S)
        fr, P = psd(r, 1.0 / DT_S)
        d_raw, f_raw, _ = prominence(fr, P, GAMMA)
        d_ntc, f_ntc, _ = prominence_notched(fr, P, GAMMA, DRIVE_HZ)
        if gh is None:
            ok = "n/a (no rhythm exists)"
        else:
            ok = ("FOUND" if abs(f_ntc - gh) < 1.0 else "*** MISSED ***") \
                 + ("   raw MISSED it" if abs(f_raw - gh) >= 1.0 else "")
        print(f"  {label:>12} | {d_raw:>7.2f} {f_raw:>8.1f} Hz | "
              f"{d_ntc:>8.2f} {f_ntc:>9.1f} Hz | {ok}")
    print()
    print("  The raw peak lands at 30.0 Hz in all four cases. Whether a rhythm")
    print("  exists, and at what frequency, changes its answer not at all.")
    print()


N_NEG = 40      # datasets per negative case. NOT 6. See below.
N_POS = 20      # datasets per positive case
FP_MAX = 4      # of 40. Derived: under Binomial(40, alpha=0.05), P(X > 4) = 0.048,
                # and 4 is the smallest threshold with a tail probability below
                # 0.05.  >>> [1 - binom.cdf(k, 40, 0.05) for k in (3,4,5)]
                #         [0.138, 0.048, 0.014]



def e7_validate_the_repair():
    """Manuscript section 3.6. The only claim that licenses using the notched
    statistic on the network at all.

    ACCEPTANCE CRITERION. A test is validated by its error rate over many
    datasets, never by a single draw, so the negative cases run 40 datasets each
    and the criterion is that the false-positive count is CONSISTENT WITH alpha,
    not that it is zero. Demanding zero false positives over 6 draws is not a
    test: a correctly calibrated 5% test passes it only 0.95**6 = 74% of the time.
    See FP_MAX above for the derivation of the threshold."""
    print("=" * 84)
    print("E7  VALIDATION OF THE REPAIR ON GROUND TRUTH")
    print("=" * 84)
    n_surr = 500
    neg = [
        ("no rhythm (sigma=25 ms)", dict(theta_jit=25.0, gamma_hz=None)),
        ("tight sigma=8, no gamma", dict(theta_jit=8.0,  gamma_hz=None)),
        ("tight sigma=3, no gamma", dict(theta_jit=3.0,  gamma_hz=None)),
    ]
    pos = [
        ("tight + real 40 Hz", dict(theta_jit=5.0, gamma_hz=40.0), 40.0),
        ("tight + real 55 Hz", dict(theta_jit=5.0, gamma_hz=55.0), 55.0),
        ("tight + real 68 Hz", dict(theta_jit=5.0, gamma_hz=68.0), 68.0),
    ]
    print(f"  Comb-notched. N_surr = {n_surr}. {N_NEG} datasets per negative case,")
    print(f"  {N_POS} per positive case. A test is validated by its error rate,")
    print(f"  never by a single draw. Accept iff false positives <= {FP_MAX}/{N_NEG}")
    print(f"  (binomial, alpha = 0.05) AND power = 100%.\n")
    print(f"  {'case':<24}{'truth':>8}{'prom':>9}{'p':>8}{'peak':>9}"
          f"{'errors':>12}  VERDICT")
    print("  " + "-" * 78)
    fails = 0

    def run(kw, n, base):
        return [test_rhythm(ground_truth(T_S, np.random.default_rng(base + k), **kw),
                            N_NEURON, T_S, band=GAMMA, n_surr=n_surr,
                            method="jitter", seed=k, drive_hz=DRIVE_HZ, notch=True)
                for k in range(n)]

    for name, kw in neg:
        rs = run(kw, N_NEG, 7000)
        fp = sum(r["is_rhythm"] for r in rs)
        good = fp <= FP_MAX
        fails += (not good)
        print(f"  {name:<24}{'none':>8}"
              f"{np.mean([r['obs_db'] for r in rs]):>6.2f} dB"
              f"{np.median([r['p'] for r in rs]):>8.3f}"
              f"{np.median([r['fpk'] for r in rs]):>6.1f} Hz"
              f"{f'{fp}/{N_NEG} FP':>12}  "
              f"{'correctly rejected' if good else '*** TOO MANY FALSE POSITIVES ***'}")

    for name, kw, truth in pos:
        rs = run(kw, N_POS, 8000)
        found = sum(r["is_rhythm"] and abs(r["fpk"] - truth) < 1.0 for r in rs)
        good = found == N_POS
        fails += (not good)
        print(f"  {name:<24}{truth:>6.0f} Hz"
              f"{np.mean([r['obs_db'] for r in rs]):>6.2f} dB"
              f"{np.median([r['p'] for r in rs]):>8.3f}"
              f"{np.median([r['fpk'] for r in rs]):>6.1f} Hz"
              f"{f'{N_POS - found}/{N_POS} FN':>12}  "
              f"{'correctly found' if good else '*** MISSED ***'}")
    print()
    if fails:
        print(f"  *** {fails} OF 6 CASES FAILED. The repair is NOT validated. ***")
    else:
        print("  Six of six. The false-positive rate is at nominal alpha, including")
        print("  in the worst case (3 ms locking, where the RAW statistic is wrong")
        print("  100% of the time). Power is 100%, and every rhythm is recovered to")
        print("  within 1 Hz of its true frequency.")
    print()
    return fails


if __name__ == "__main__":
    v = e1_null_of_statistic()
    e2_theta_harmonics(v)
    e3_screens_cannot_rescue_a_masked_rhythm()
    e4_synchrony_defeats_the_null()
    e5_peak_is_a_comb_not_a_band_edge()
    e6_the_comb_masks_real_rhythms()
    fails = e7_validate_the_repair()
    print("=" * 84)
    print("No network was simulated in this file. These are properties of the")
    print("STATISTIC and of THETA-LOCKED SPIKING, not of any one implementation.")
    print("=" * 84)
    if fails:
        raise SystemExit(f"E7 FAILED IN {fails} CASES -- do not report these results")
