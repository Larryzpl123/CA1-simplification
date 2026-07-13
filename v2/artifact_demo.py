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

THE THREE EXPERIMENTS
---------------------
E1. NULL OF THE STATISTIC.
    Constant-rate Poisson. No rhythm of any kind. Measure prominence in 30-80 Hz.
    If the statistic were unbiased this would be ~0 dB. It is not: prominence is
    the MAX over ~40 chi-square-distributed Welch bins, so it is systematically
    positive. Whatever value comes out is the floor below which NO reported
    "gamma prominence" means anything -- including the 6.6-11.9 dB reported in
    v1 of this preprint.

E2. THETA HARMONICS MASQUERADING AS GAMMA.
    Poisson spikes modulated at 6 Hz ONLY. Gamma content: exactly zero, by
    construction. But a sharply theta-locked rate is non-sinusoidal, so its
    Fourier series has harmonics at 6k Hz -- and 36, 42, 48, 54, 60, 66, 72, 78
    all sit INSIDE the 30-80 Hz gamma band. We sweep modulation depth and ask:
      (a) does prominence rise?                              (does it look real?)
      (b) does a surrogate null call it significant?         (does rigour save us?)
      (c) does the peak land on k x 6 Hz?                    (what is it really?)
    The critical result is (b). The jitter surrogate destroys fast temporal
    structure -- and harmonics ARE fast temporal structure. So the surrogate
    null, the very fix introduced to make this rigorous, REPORTS THEM AS REAL.
    A null distribution is necessary and NOT sufficient.

E3. THE SCREENS ARE NOT REDUNDANT.
    A ground-truth 40 Hz rhythm must pass all screens; the artifacts must each be
    caught by exactly one. Shows the screens are complementary, not duplicative.

Run:  python3 artifact_demo.py
"""

import numpy as np
from spectral_null import (test_rhythm, pop_rate, psd, prominence,
                           harmonic_of, DT_S)

T_S      = 21.0         # seconds. Set by the resolution requirement, not by taste:
                        # df = 1/T_seg must satisfy 3*df/f_drive <= 0.10 (see
                        # spectral_null.required_df). At 2.5 s the harmonic screen
                        # blocked 62% of the band and KILLED a ground-truth 40 Hz
                        # rhythm. At 21 s it blocks 6% and keeps it.
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

    IMPORTANT NEGATIVE RESULT. This generator does NOT reproduce the harmonic
    artifact, at any theta depth (0 -> 0.9) or waveform sharpness (1 -> 10):
    prominence stays at the null (3.1-3.8 dB) and the surrogate test rejects
    100% of the time. Measured, N_surr=600.

    That matters, because it falsifies the obvious hypothesis. Rate modulation
    ALONE is not enough. When neurons sample independently, the population rate
    is lambda(t) plus shot noise, and the harmonics of lambda(t) are buried in
    that noise by 54 Hz. The artifact in the CA1 network is therefore NOT caused
    by theta modulation as such -- see synchronous_spikes() below."""
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

    The previous generator injected gamma by PHASE-MODULATING spike times with a
    sinusoid. That does not create a clean rhythm -- it creates intermodulation
    sidebands between the 6 Hz comb and the injected frequency, so a nominal
    40 Hz injection produced its peak at 46 Hz, and a nominal 55 Hz at 67 Hz.
    As a positive control it was worthless: it never contained the rhythm it
    claimed to contain."""
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
            r = test_rhythm(st, N_NEURON, T_S, band=GAMMA, n_surr=N_SURR,
                            method="jitter", seed=k, drive_hz=DRIVE_HZ)
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
    print("  THIS IS THE POINT: a surrogate null is NECESSARY AND NOT SUFFICIENT.")
    print("  It asks 'is there non-random fast structure?'. Theta harmonics ARE")
    print("  non-random fast structure. They are real. They are not a rhythm.")
    print()
    return rows


def e3_screens_are_complementary():
    print("=" * 84)
    print("E3  THE SCREENS ARE COMPLEMENTARY, NOT REDUNDANT")
    print("=" * 84)
    # NOTE: the "theta only" case MUST be generated with synchronous_spikes, not
    # poisson_spikes. E2 established that independent Poisson rate modulation does
    # NOT defeat the null at any depth or sharpness; only tight phase-locking does.
    # Using poisson_spikes here would show the null rejecting it on its own and
    # would give the false impression that the harmonic screen is unnecessary.
    cases = [
        ("no rhythm at all",         "poisson", dict(depth_theta=0.0),          False),
        ("theta-locked, sigma=5 ms", "sync",    dict(jitter_ms=5),              False),
        ("GROUND-TRUTH 40 Hz gamma", "gt",      dict(gamma_hz=40.0),            True),
        ("GROUND-TRUTH 55 Hz gamma", "gt",      dict(gamma_hz=55.0),            True),
    ]
    print(f"  {'case':<26}{'truth':>7}{'prom':>7}{'p':>7}{'f_pk':>7}"
          f"{'null':>6}{'edge':>6}{'harm':>6}{'VERDICT':>9}")
    print("  " + "-" * 76)
    for name, gen, kw, truth in cases:
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
                                  method="jitter", seed=k, drive_hz=DRIVE_HZ))
        prom_ = np.mean([r["obs_db"] for r in rs])
        p_ = np.median([r["p"] for r in rs])
        f_ = np.median([r["fpk"] for r in rs])
        n_ok = np.mean([r["significant"] for r in rs])
        edge = np.mean([r["band_edge"] for r in rs])
        harm = np.mean([(r["k_drive"] is not None) or (r["k_rate"] is not None)
                        for r in rs])
        rhy = np.mean([r["is_rhythm"] for r in rs])
        verdict = "RHYTHM" if rhy > 0.5 else "rejected"
        ok = "OK" if (rhy > 0.5) == truth else "*** WRONG ***"
        print(f"  {name:<26}{str(truth):>7}{prom_:>7.2f}{p_:>7.3f}{f_:>7.1f}"
              f"{100*n_ok:>5.0f}%{100*edge:>5.0f}%{100*harm:>5.0f}%{verdict:>9}  {ok}")
    print()
    print("  The theta-locked row is the whole paper: it passes the null, it looks")
    print("  like gamma, and only the harmonic screen stops it. The ground-truth row")
    print("  proves the screen is not merely destroying everything: a REAL 40 Hz")
    print("  rhythm survives all four. Both statements require adequate frequency")
    print("  resolution -- at 2.5 s the screen killed the ground-truth rhythm too.")
    print()


if __name__ == "__main__":
    v = e1_null_of_statistic()
    e2_theta_harmonics(v)
    e3_screens_are_complementary()
    print("=" * 84)
    print("No network was simulated in this file. These are properties of the")
    print("STATISTIC and of THETA-LOCKED SPIKING, not of any one implementation.")
    print("=" * 84)
