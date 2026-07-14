#!/usr/bin/env python3
"""
ping_scaling_test.py — the only check on the list that is an EXPERIMENT.
Larry (Peilin) Zhong

WHY THIS FILE EXISTS
--------------------
Every other check in this project can only REJECT. The surrogate null rejects
peaks that are indistinguishable from noise. The band-edge flag rejects fitting
artifacts. The harmonic screens reject peaks sitting on multiples of the drive or
of the firing rate. The comb notch removes the drive's harmonics from the
spectrum entirely, before the maximum is taken.

In ca1_v2.py, two of five conditions produce a gamma-band peak that survives ALL
of that. The null is cleared (p = 0.011 and p = 0.001). It is not at a band edge.
It cannot be a drive harmonic, because those bins are no longer in the spectrum.

Every analysis-side check is exhausted, and a significant peak remains.

Nothing you can do to that spectrum will tell you what the peak is. To find out,
you have to perturb the mechanism that is supposed to have generated it, and see
whether the peak responds the way that mechanism requires.

THE PREDICTION
--------------
In PING (pyramidal-interneuron network gamma), the cycle period is set by the
decay of inhibition: excitation recruits inhibition, inhibition silences
excitation, and the next cycle begins when inhibition has decayed. The peak
frequency therefore falls roughly as 1/tau_GABA.

A harmonic of the drive does not care about tau_GABA. Neither does a statistical
fluctuation. Neither does a resonance set by some other time constant.

So: sweep tau_GABA over a wide range and watch the peak.

    peak moves ~ 1/tau_GABA   ->  PING. This is a rhythm.
    peak does not move        ->  NOT PING. Whatever it is, it is not the
                                  excitatory-inhibitory loop oscillating.

THE TEST STATISTIC -- AND A THRESHOLD I GOT WRONG
-------------------------------------------------
The criterion is on the PERIOD, not the frequency. PING predicts
    T_cycle = fixed_delay + k * tau_GABA
which is affine in tau, so a 12-fold change in tau moves the FREQUENCY only about
2-fold. A criterion that thresholds on frequency change therefore rejects a
network that is obeying the theory.

PING does not predict f = 1/(2*tau_GABA). The cycle is: E fires -> recruits I ->
I inhibits E -> E recovers when inhibition decays. Only the LAST step scales with
tau_GABA. The others (AMPA rise/decay, membrane time constant, refractory period)
are FIXED. So the prediction is on the PERIOD, and it is affine:

        T_cycle  =  (fixed loop delay)  +  k * tau_GABA

A tenfold change in tau_GABA therefore moves the frequency by much LESS than
tenfold, and a ratio threshold on frequency will reject genuine PING. Measured
on the Scaled 4x condition:

    tau_GABA    peak      period       fit: T = 19.1 + 0.673 * tau   (R2 = 0.81)
      2 ms     55.2 Hz    18.1 ms      the 19.1 ms intercept IS the fixed delay
      5 ms     43.5 Hz    23.0 ms      (tau_AMPA = 5 ms + membrane + refractory)
     10 ms     34.3 Hz    29.2 ms
     20 ms     32.3 Hz    31.0 ms

Frequency moved only 1.71x, and the old threshold called that "not PING". It is
exactly what PING predicts. An unjustified hard threshold gave the wrong answer
in a study whose entire subject is unjustified hard thresholds.

AND A SECOND THRESHOLD I ALSO GOT WRONG
---------------------------------------
The second version added a MONOTONICITY gate on top of the regression: the peak
frequency had to fall at every step of the sweep. On 8 tau values with 3 seeds
each and peak estimates scattering by +/-10 Hz, that is very nearly impossible to
satisfy even when the effect is real. It duly vetoed a slope that was positive,
large, and significant (Scaled 4x: +0.539 +/- 0.176, R2 = 0.61, p = 0.022).

Monotonicity was never a prediction of PING. PING predicts an affine relation
between period and tau_GABA; the regression IS that prediction. The monotonicity
gate was an extra hard threshold that I bolted on, and it did what unjustified
hard thresholds do.

That makes TWO thresholds, in two successive versions, each of which rejected the
same significant result. Which raises a problem I cannot analyse my way out of:

    I changed the acceptance criterion twice, and each change moved the verdict
    closer to "PING". That is p-hacking, however good the reasons sound.

PRE-REGISTRATION
----------------
The criterion below is now FIXED, and it is derived from the mechanism, not from
the data:

    PING predicts   T_cycle = fixed_loop_delay + k * tau_GABA     (affine)
    TEST:           regress period on tau_GABA
    ACCEPT iff      slope > 0  AND  p < 0.05
    No other gate. No monotonicity requirement. No ratio threshold.

It is then run on SEEDS NEVER PREVIOUSLY USED (200-204, five of them rather than
three, because the per-seed peak estimates are noisy). The seeds used while the
criterion was being changed (100-102) are burned and are not used again.

If the slope survives on fresh seeds under a criterion fixed in advance, the
result stands. If it does not, it dies, and it should.

WHAT THIS MEANS BEYOND THIS MODEL
---------------------------------
A spectral analysis, however careful, cannot by itself establish that a peak is a
network rhythm. Screens remove candidates; they never confirm one. At some point
the claim has to be made falsifiable by perturbing the mechanism it names.

That step is not a statistic. It is an experiment.

Run:  python3 ping_scaling_test.py
"""

import sys
import numpy as np
from scipy import stats

sys.path.insert(0, ".")
import ca1_v2 as V
from brian2 import ms, second
from spectral_null import test_rhythm, print_version_banner, DT_S, NPERSEG

# tau_GABA values to sweep. The range must be WIDE -- an order of magnitude --
# because a weak dependence is not evidence of PING, and a narrow sweep cannot
# distinguish "moves a little" from "does not move".
# 8 values, log-spaced-ish, spanning 15x. Four points gave R2 = 0.81 but p = 0.098
# on the Scaled 4x condition -- a strong trend that n could not certify. The claim
# lives or dies on this grid, so it is not the place to economise on compute.
TAU_GRID = [2, 3, 4, 6, 8, 12, 16, 24]   # ms
# FRESH SEEDS. 100-102 were used while the acceptance criterion was still being
# changed, so they are burned: any result on them is contaminated by researcher
# degrees of freedom. These five have never been run.
SEEDS = [200, 201, 202, 203, 204]
N_SURR = 1000
DRIVE_HZ = 6.0

CONDITIONS = [
    # name,          n_exc, n_inh, p_ie
    ("LIF baseline",    80,   20,  V.P_IE_BASE),
    ("Scaled 4x",      320,   80,  V.P_IE_BASE),
]


def main():
    df = (1.0 / DT_S) / NPERSEG
    print_version_banner({
        "freq resolution df": f"{df:.3f} Hz",
        "recording": f"{V.DURATION/second:.0f} s",
        "comb notch": "ON",
        "tau_GABA sweep": f"{TAU_GRID} ms",
        "N_surr": N_SURR,
    })
    print()
    print("=" * 88)
    print("PING SCALING TEST — the check that is an experiment, not a statistic")
    print("=" * 88)
    print("PING requires the peak frequency to fall roughly as 1/tau_GABA, because")
    print("inhibitory decay sets the cycle period. A drive harmonic, a statistical")
    print("fluctuation, and a resonance set by any other time constant will not move.")
    print()

    for name, ne, ni, pie in CONDITIONS:
        print("-" * 88)
        print(f"{name}")
        print("-" * 88)
        print(f"  {'tau_GABA':>9}{'rate_E':>9}{'rate_I':>9}{'gamma p':>10}"
              f"{'prom':>8}{'null':>7}{'PEAK':>8}   per-seed peaks")
        rows = []
        for tg in TAU_GRID:
            ps, fs, pr, nl, re_, ri_ = [], [], [], [], [], []
            for sd in SEEDS:
                r = V.build_and_run(ne, ni, int_model="lif", tau_gaba=tg * ms,
                                    p_ie=pie, seed_val=sd, normalize=True)
                g = test_rhythm(r["spikes_e"], r["n_exc"], r["T"], band=(30, 80),
                                n_surr=N_SURR, method="jitter", seed=sd,
                                drive_hz=DRIVE_HZ)
                ps.append(g["p"]); fs.append(g["fpk"]); pr.append(g["obs_db"])
                nl.append(g["null_mean"])
                re_.append(r["rate_e"]); ri_.append(r["rate_i"])
            f_med = float(np.median(fs))
            rows.append((tg, f_med, float(np.median(ps))))
            print(f"  {tg:>7}ms{np.mean(re_):>9.1f}{np.mean(ri_):>9.1f}"
                  f"{np.median(ps):>10.3f}{np.mean(pr):>8.2f}{np.mean(nl):>7.2f}"
                  f"{f_med:>8.1f}   {[round(x,1) for x in fs]}", flush=True)

        # ---- the verdict: regress PERIOD on tau_GABA ----
        # NOT a ratio threshold on frequency. See the header: PING predicts
        # T = fixed_delay + k*tau, so frequency moves far less than tau does, and
        # a ratio threshold rejects real PING. This is the corrected test.
        # PRE-REGISTERED, and it is the whole criterion. Nothing is bolted on.
        #     PING predicts  T = fixed_delay + k*tau_GABA
        #     ACCEPT iff     slope > 0  AND  p < 0.05
        tau = np.array([r[0] for r in rows], float)
        fpk = np.array([r[1] for r in rows], float)
        T = 1000.0 / fpk                              # period, ms
        reg = stats.linregress(tau, T)

        print()
        print(f"  PING prediction:  period = fixed_delay + k * tau_GABA   (affine)")
        print(f"  fitted:           period = {reg.intercept:.1f} + "
              f"{reg.slope:.3f} * tau_GABA   (ms)")
        print(f"  slope = {reg.slope:+.3f} +/- {reg.stderr:.3f}   "
              f"R^2 = {reg.rvalue**2:.3f}   p = {reg.pvalue:.4f}")
        print(f"  PRE-REGISTERED CRITERION: slope > 0 AND p < 0.05. Nothing else.")
        print()
        if reg.slope > 0 and reg.pvalue < 0.05:
            print("  => CRITERION MET. The cycle period lengthens with the inhibitory decay")
            print("     constant, as PING requires. The peak is a network rhythm generated")
            print(f"     by the E-I loop, and the {reg.intercept:.0f} ms intercept is the fixed part of")
            print("     the loop delay (AMPA, membrane, refractory) -- which is why the")
            print("     FREQUENCY moves far less than tau does.")
        else:
            print("  => CRITERION NOT MET. NOT PING. Whatever this peak is, it is not the")
            print("     excitatory-inhibitory loop oscillating, and no claim of emergent")
            print("     gamma is warranted -- however many screens it passed.")
            print("     Screens reject; only this test confirms.")
        print()

    print("=" * 88)
    print("Screens can only reject. This is the only check that can confirm — and it")
    print("is the only one that requires re-running the model rather than re-analysing")
    print("a spectrum. That is not an inconvenience. That is the point.")
    print("=" * 88)


if __name__ == "__main__":
    main()
