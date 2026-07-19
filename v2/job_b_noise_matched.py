#!/usr/bin/env python3
"""
job_b_noise_matched.py -- separate network SIZE from input FLUCTUATION.

Version 3 of the paper said "network size, not reduction, is the switch": the
320+80 network passes the pre-registered PING test and the 80+20 one does not.
Version 4 withdrew that, because the two networks differ in TWO things at once.
The weight normalisation in ca1_v2.scaled_weights() holds the MEAN synaptic
conductance per cell constant as N changes, but not its FLUCTUATIONS. For a
pathway with presynaptic count N and connection probability p, the coefficient
of variation of the total conductance onto a postsynaptic cell is

    CV = sqrt( (1 - p) / (p * N) )                    (Borgers 2017, Sec. 30.3)

At the base connection probabilities the 80+20 network sits at TWICE the CV of
the 320+80 network on every pathway, on an I->E in-degree of five. So the failing
80+20 result could be about the cell count OR about the noise, and the paper as
run cannot tell them apart.

This script holds the CV fixed and lets only N change, which is the comparison
Brunel & Wang (2003) use to isolate N. It raises the 80+20 network's connection
probabilities to the values that reproduce the 320+80 network's per-pathway CV,
solved from Borgers's formula, and runs the pre-registered PING test on it,
against the 320+80 network unchanged.

    If the noise-matched 80+20 network now PASSES the PING test, the switch was
    the fluctuation and not the count, and Borgers's "even one E-cell and one
    I-cell" remark is the limit of the same statement.

    If it still fails, the count is doing something the noise is not, and that
    is a result too.

NOTHING here modifies a gated file. It imports ca1_v2.build_and_run and
ping_scaling_test.test_rhythm and calls them with different connection
probabilities; the computation inside them is byte-identical to the archived
versions. This file is not in run_all.sh's hash gate for that reason: it changes
no gated result, it asks a new question of the gated code.

    cd github_upload/v2
    python3 job_b_noise_matched.py 2>&1 | tee ../results/jobB_$(date +%Y%m%d_%H%M%S).log
"""
import numpy as np
from math import sqrt
from scipy import stats
from brian2 import ms, second

import ca1_v2 as V
from ping_scaling_test import test_rhythm, TAU_GRID, NPERSEG, DT_S, SEEDS, N_SURR, DRIVE_HZ


# ---------------------------------------------------------------------------
# The noise-matching arithmetic, and the assertion that it is correct.
# ---------------------------------------------------------------------------
def cv(p, n_pre):
    """Coefficient of variation of the total conductance, Borgers 2017 Sec 30.3."""
    return sqrt((1.0 - p) / (p * n_pre))


def p_matching(cv_target, n_pre_small):
    """The p that makes CV(p, n_pre_small) == cv_target. Inverts the formula:
       (1-p)/(p*N) = cv^2  ->  p = 1/(1 + cv^2 * N)."""
    return 1.0 / (1.0 + cv_target * cv_target * n_pre_small)


# The BIG network fixes the target CVs. Which presynaptic population feeds each
# pathway: E->E and E->I are fed by the 320 excitatory cells; I->E and I->I by
# the 80 inhibitory cells. The SMALL network has 80 exc and 20 inh.
BIG   = dict(n_exc=320, n_inh=80)
SMALL = dict(n_exc=80,  n_inh=20)
BASE  = dict(p_ee=V.P_EE_BASE, p_ei=V.P_EI_BASE, p_ie=V.P_IE_BASE, p_ii=V.P_II_BASE)

_pre_big   = dict(p_ee=BIG["n_exc"],   p_ei=BIG["n_exc"],
                  p_ie=BIG["n_inh"],   p_ii=BIG["n_inh"])
_pre_small = dict(p_ee=SMALL["n_exc"], p_ei=SMALL["n_exc"],
                  p_ie=SMALL["n_inh"], p_ii=SMALL["n_inh"])

CV_TARGET = {k: cv(BASE[k], _pre_big[k])   for k in BASE}
P_SMALL   = {k: p_matching(CV_TARGET[k], _pre_small[k]) for k in BASE}

# SELF-CHECK. A wrong p here would produce a clean log with a silently wrong
# experiment, which is the failure this whole paper is about. So the noise match
# is asserted before a single network is built: every matched pathway's CV must
# equal the big network's to 1e-9, and every matched probability must be a real
# probability strictly below 1 (no all-to-all).
print("=" * 78)
print("NOISE MATCH — solved from Borgers (2017), Sec. 30.3, and checked here")
print("=" * 78)
print(f"  {'pathway':7} {'N_big':>6} {'p_big':>7} {'CV_big':>8}  |  "
      f"{'N_sml':>6} {'p_small':>8} {'CV_small':>9}")
for k in ("p_ee", "p_ei", "p_ie", "p_ii"):
    cb, cs = CV_TARGET[k], cv(P_SMALL[k], _pre_small[k])
    assert abs(cb - cs) < 1e-9, f"{k}: CV not matched ({cb} vs {cs})"
    assert 0.0 < P_SMALL[k] < 1.0, f"{k}: p_small={P_SMALL[k]} is not a probability below 1"
    print(f"  {k:7} {_pre_big[k]:>6} {BASE[k]:>7} {cb:>8.4f}  |  "
          f"{_pre_small[k]:>6} {P_SMALL[k]:>8.4f} {cs:>9.4f}")
print("  every pathway matched to 1e-9, every p_small in (0,1). No all-to-all.")
print("=" * 78)


# ---------------------------------------------------------------------------
# The two conditions. Same pre-registered test as ping_scaling_test.py, same
# fresh seeds (200-204), same criterion: slope k > 0 AND p < 0.05.
# ---------------------------------------------------------------------------
CONDITIONS = [
    # name,                         n_exc, n_inh, p_kwargs
    ("320+80  (control, base p)",   320, 80, dict(BASE)),
    ("80+20   (noise-matched p)",    80, 20, dict(P_SMALL)),
]


def run_condition(name, ne, ni, pk):
    print("-" * 78)
    print(name)
    print(f"  p_ee={pk['p_ee']:.4f}  p_ei={pk['p_ei']:.4f}  "
          f"p_ie={pk['p_ie']:.4f}  p_ii={pk['p_ii']:.4f}")
    print("-" * 78)
    print(f"  {'tau_GABA':>9}{'rate_E':>9}{'rate_I':>9}{'gamma p':>10}"
          f"{'PEAK':>8}   per-seed peaks")
    rows = []
    for tg in TAU_GRID:
        ps, fs, re_, ri_ = [], [], [], []
        for sd in SEEDS:
            r = V.build_and_run(ne, ni, int_model="lif", tau_gaba=tg * ms,
                                p_ie=pk["p_ie"], p_ee=pk["p_ee"],
                                p_ei=pk["p_ei"], p_ii=pk["p_ii"],
                                seed_val=sd, normalize=True)
            g = test_rhythm(r["spikes_e"], r["n_exc"], r["T"], band=(30, 80),
                            n_surr=N_SURR, method="jitter", seed=sd,
                            drive_hz=DRIVE_HZ)
            ps.append(g["p"]); fs.append(g["fpk"])
            re_.append(r["rate_e"]); ri_.append(r["rate_i"])
        f_med = float(np.median(fs))
        rows.append((tg, f_med))
        print(f"  {tg:>7}ms{np.mean(re_):>9.1f}{np.mean(ri_):>9.1f}"
              f"{np.median(ps):>10.3f}{f_med:>8.1f}   "
              f"{[round(x, 1) for x in fs]}", flush=True)

    tau = np.array([a for a, _ in rows], float)
    T = 1000.0 / np.array([b for _, b in rows], float)
    reg = stats.linregress(tau, T)
    passed = reg.slope > 0 and reg.pvalue < 0.05
    print()
    print(f"  fitted: period = {reg.intercept:.1f} + {reg.slope:.3f} * tau_GABA (ms)")
    print(f"  slope = {reg.slope:+.3f} +/- {reg.stderr:.3f}   "
          f"R^2 = {reg.rvalue**2:.3f}   p = {reg.pvalue:.4f}")
    print(f"  PRE-REGISTERED CRITERION (slope > 0 AND p < 0.05): "
          f"{'PASS -- PING' if passed else 'FAIL -- not PING'}")
    print()
    return passed


def main():
    V.print_version_banner({
        "recording": f"{V.DURATION/second:.0f} s",
        "comb notch": "ON",
        "tau_GABA sweep": f"{TAU_GRID} ms",
        "N_surr": N_SURR,
        "seeds": f"{SEEDS}",
    })
    print()
    print("=" * 78)
    print("JOB B — is the switch the cell COUNT, or the input FLUCTUATION?")
    print("=" * 78)
    print("The 320+80 control is unchanged and must still PASS (that is v4's result).")
    print("The 80+20 network is run at connection probabilities that match its per-")
    print("pathway CV to the 320+80 network's, so that only N differs. If it now")
    print("passes too, the switch was the noise and not the count.")
    print()

    results = {}
    for name, ne, ni, pk in CONDITIONS:
        results[name] = run_condition(name, ne, ni, pk)

    print("=" * 78)
    ctrl = [k for k in results if k.startswith("320")][0]
    small = [k for k in results if k.startswith("80")][0]
    assert results[ctrl], (
        "The 320+80 control FAILED the PING test. It passed in v4 with identical "
        "code, so something in the environment differs. Do not interpret the "
        "80+20 result until the control passes.")
    if results[small]:
        print("RESULT: noise-matched 80+20 PASSES. With the fluctuations held equal")
        print("to the 320+80 network's, the small network is PING after all. The")
        print("switch was the input noise, not the cell count. This is the limit of")
        print("Borgers's one-E-one-I remark, reached from the other side.")
    else:
        print("RESULT: noise-matched 80+20 still FAILS. Matching the fluctuations was")
        print("not enough to make the small network PING, so the cell count is doing")
        print("something the noise is not. What that is remains open.")
    print("=" * 78)


if __name__ == "__main__":
    main()
