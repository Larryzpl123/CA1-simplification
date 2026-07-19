#!/usr/bin/env python3
"""
ca1_v2.py -- corrected replacement for the v1 preprint
(v1 preprint source file: CA1-simplification-v3.py -- "v3" there is the third CODE
draft, not a third preprint. There is one prior preprint: v1.)
Larry (Peilin) Zhong

WHAT WAS WRONG IN v1 (all three verified by re-running v1's own code)
---------------------------------------------------------------------
1. ONE connection probability for ALL connection types:
       conn = (rng.random((N,N)) < p_conn)      # p_conn = 0.02
   For the 10-neuron nets, across v1's own seeds 100-104:
       inhibitory -> excitatory synapses : [0, 0, 0, 0, 0]
       TOTAL synapses in the whole net   : [1, 0, 0, 3, 0]
   So the "microcircuit" was 10 UNCOUPLED neurons sharing a 6 Hz drive.
   -> "sparsification preserves dynamics" was vacuous (0 -> 0 connections).
   -> "no emergent gamma" was true but uninformative (no circuit to emerge from).
   -> "theta preserved under reduction" was trivial (theta was in the input, and
      there was no network that could have destroyed it).

2. SILENT INTERNEURONS in the hybrid net. make_net passed I_mean_lif = 1.0 to
   the HH interneurons, but the HH rheobase is ~6-7 uA/cm^2 (measured: I=1.0 ->
   0 sustained spikes; I=7.0 -> 46). The hybrid arm, whose whole purpose
   was to test "does retaining detailed inhibitory kinetics recover gamma?",
   contained interneurons that never fired once.

3. NO NULL DISTRIBUTION for the peak-prominence statistic. Prominence takes the
   MAX over ~40 chi-square-distributed Welch bins, so it is positively biased:
   on data with NO rhythm the statistic returns 5.62 +/- 0.61 dB at the analysis
   settings used here (200 no-rhythm datasets; artifact_demo E1). The null also
   MOVES with the Welch segment length. v1 reported theta prominence of 6.6-7.5 dB
   as evidence of a preserved rhythm. That is close to the noise floor of the
   statistic itself. v1's band-edge and harmonic flags do not catch this.

WHAT v2 DOES DIFFERENTLY
------------------------
  * Synapses declared PER TYPE (E->E, E->I, I->E, I->I). Making v1's bug requires
    deliberately typing the same probability four times.
  * PRE-FLIGHT ASSERTIONS, before any science runs:
        - every synapse type has > 0 synapses
        - every population actually fires (rate > 0)
    These two lines are what v1 needed.
  * ONE consistent current unit for HH and LIF, with SEPARATELY NAMED drives
    (I_DRIVE_PYR / I_DRIVE_INT_HH / I_DRIVE_INT_LIF). Accidentally feeding a
    LIF-scaled current to an HH cell is now a NameError, not a silent zero.
  * Rhythms are tested against a SURROGATE NULL (spectral_null.py; validated at
    2/40 false positives -- the nominal 5% -- and 20/20 power). Report p, not dB.
    A surrogate null is NECESSARY AND NOT SUFFICIENT: the drive's harmonics are
    non-random fast structure, so it calls them significant. The comb is notched
    out of the spectrum BEFORE the maximum is taken. See artifact_demo E2, E3.
  * v1's one valid arm — single-neuron HH vs LIF cost — is retained, with all
    models on the same integrator and dt, as v1 correctly insisted.

REQUIRES:  pip install brian2 numpy scipy
           spectral_null.py in the same folder
"""

import time
import numpy as np
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, PopulationRateMonitor, Network,
    defaultclock, prefs, start_scope, seed as brian_seed,
    ms, mV, nS, pA, uF, cm, umetre, siemens, msiemens, ufarad, second, Hz, volt,
)
from spectral_null import (test_rhythm, print_version_banner, blocked_fraction,
                           DT_S, NPERSEG)

prefs.codegen.target = "numpy"

# ============================================================================
# BIOPHYSICS — one unit system for BOTH models (this is the anti-v1 measure)
# ============================================================================
AREA   = 20000 * umetre ** 2
C_M    = (1 * ufarad * cm ** -2) * AREA
G_L    = (5e-5 * siemens * cm ** -2) * AREA
E_L    = -65 * mV
TAU_M  = C_M / G_L                       # ~20 ms, and it is DERIVED, not asserted

# HH channels (Traub-Miles)
E_NA, E_K = 50 * mV, -90 * mV
G_NA = (100 * msiemens * cm ** -2) * AREA
G_KD = (30 * msiemens * cm ** -2) * AREA
V_T  = -63 * mV

# LIF
V_TH, V_RESET = -50 * mV, -65 * mV
REFRAC_PYR, REFRAC_INT = 2 * ms, 1 * ms

# synapses
E_EXC, E_INH = 0 * mV, -80 * mV
TAU_AMPA = 5 * ms
TAU_GABA_DEFAULT = 5 * ms

# ---- drives: SEPARATELY NAMED. A LIF current can never reach an HH cell. ----
F_THETA        = 6 * Hz
I_DRIVE_PYR    = 190 * pA      # suprathreshold for the LIF pyramidal
I_DRIVE_INT_LIF = 120 * pA     # SUBthreshold: I-cells must be recruited by E
I_DRIVE_INT_HH = 240 * pA      # ABOVE HH rheobase (v1's bug: it used the LIF value)
I_THETA        = 30 * pA
SIGMA_V        = 3 * mV

# ---- weights, specified AT THE REFERENCE NETWORK SIZE ----
#
# W_IE = 6 nS is adopted because it yields a pyramidal firing rate of ~6 Hz, the
# physiological target for CA1 pyramidal cells. It is chosen on the RATE.
#
# It is NOT selected by the positive control. calibrate_w_ie() below runs the
# sweep and prints what it finds: theta is recovered in 100% of seeds at every
# weight tested, so the control establishes that the pipeline HAS POWER and
# nothing more. It never examines the gamma band, so it cannot bias the gamma
# result in either direction.
W_IE_ADOPTED_NS = 6.0
W_EE_REF, W_EI_REF, W_IE_REF, W_II_REF = (0.4 * nS, 6.0 * nS,
                                          W_IE_ADOPTED_NS * nS, 8.0 * nS)

# connectivity — PER TYPE.  E->E stays sparse (CA1 really is); the rest are not.
P_EE_BASE, P_EI_BASE, P_IE_BASE, P_II_BASE = 0.02, 0.20, 0.25, 0.20

# reference sizes the weights above were calibrated at
N_EXC_REF, N_INH_REF = 80, 20

# ---- IN-DEGREE NORMALISATION (third bug, found while calibrating) ------------
# Without this, growing the network silently multiplies the inhibition each
# pyramidal cell receives:  in-degree = N_inh * p_ie, so 20->80 interneurons
# QUADRUPLES it. Without normalisation the "scaled 4x" arm would not test SCALE,
# it would test "4x more inhibition". Scaling w by (reference in-degree)/(actual
# in-degree) holds the MEAN total synaptic drive per cell constant.
#
# It does NOT hold "size the only thing that varies", which is what this comment
# said until 2026-07-17. Christoph Borgers asked whether the conductances were
# rescaled, and observed that if they are, the behaviour should barely depend on
# the number of cells. They are, and it does, so the scaling is not doing what
# the comment claimed. It fixes the mean and not the fluctuations: for N inputs
# of weight w the mean goes as N*w, which this pins, but the variance goes as
# N*w^2, which it divides by four. The in-degrees move with N and are not
# controlled:
#
#                      80 + 20      320 + 80
#     E->E in-degree      1.62          6.63
#     I->E in-degree      5.20         20.30
#
# Brunel & Wang (2003), cited by this paper, hold the number of connections AND
# the conductance fixed instead -- 500 cells at p = 0.4, 1,000 at p = 0.2, 2,000
# at p = 0.1, in-degree 200 throughout -- "so as to keep unchanged the temporal
# average and fluctuations of the synaptic currents as network size was varied".
# That isolates N. This does not, and so this pair of conditions cannot say
# whether N or the in-degree is what matters. The manuscript says so.
def scaled_weights(n_exc, n_inh, p_ee, p_ei, p_ie, p_ii):
    def s(w_ref, n_pre_ref, p_ref, n_pre, p):
        return w_ref * (n_pre_ref * p_ref) / max(n_pre * p, 1e-12)
    return (s(W_EE_REF, N_EXC_REF, P_EE_BASE, n_exc, p_ee),
            s(W_EI_REF, N_EXC_REF, P_EI_BASE, n_exc, p_ei),
            s(W_IE_REF, N_INH_REF, P_IE_BASE, n_inh, p_ie),
            s(W_II_REF, N_INH_REF, P_II_BASE, n_inh, p_ii))

# ---- RECORDING LENGTH IS SET BY THE HARMONIC SCREEN, NOT BY CONVENIENCE -----
# The drive-harmonic screen blocks ~3*df/f_drive of the analysis band. At 2.5 s
# (df = 0.40 Hz) that is 20% of the gamma band, and a genuine 40 Hz rhythm is
# rejected as a drive harmonic. 21 s gives df = 0.12 Hz and 6% blocked, at which
# resolution the screen rejects true harmonics and keeps true rhythms.
# Measured in v1_diagnosis.py (M2).
DURATION, TRANSIENT, DT = 21000 * ms, 1000 * ms, 0.1 * ms

# ---- N_SURR: this is pure Monte-Carlo precision on the permutation p ---------
# p_hat = (1 + #{null >= obs}) / (1 + N_surr);  SE(p_hat) ~ sqrt(p(1-p)/N_surr).
# At the decision point p = 0.05:
#     N_surr =  200  -> SE = sqrt(.05*.95/200)  = 0.0154   (p = .05 +/- .015: useless)
#     N_surr = 1000  -> SE = 0.0069
#     N_surr = 2000  -> SE = 0.0049   <- SE ~ 1/10 of alpha; this is the knee
#     N_surr = 5000  -> SE = 0.0031   (2.5x the compute for 1.6x the precision)
# 2000 is the right stopping point: beyond it, MC error is no longer what limits
# you -- NETWORK-SEED variance is (see SEEDS below). Spend compute on seeds.
N_SURR = 2000
SEEDS = [100, 101, 102, 103, 104]
GAMMA_BAND, THETA_BAND = (30.0, 80.0), (4.0, 8.0)
DRIVE_HZ = 6.0          # the imposed theta. Screen 4 checks peaks against k*DRIVE_HZ.

EQS_LIF = """
dv/dt = ( G_L*(E_L - v) + ge*(E_EXC - v) + gi*(E_INH - v)
          + I_drive + I_THETA*sin(2*pi*F_THETA*t) ) / C_M
        + SIGMA_V*xi*sqrt(2/TAU_M) : volt (unless refractory)
dge/dt = -ge/TAU_AMPA : siemens
dgi/dt = -gi/tau_gaba : siemens
I_drive : amp
"""

# NOTE: no membrane-noise term here. Brian2's exponential_euler (the standard,
# stable integrator for HH at dt=0.1 ms) cannot solve stochastic equations, and
# forward-Euler on HH at this dt is not trustworthy. The HH interneurons are
# deterministic; stochasticity reaches them through synaptic input from the noisy
# pyramidal population, which is where it belongs.
EQS_HH = """
dv/dt = ( G_NA*(m*m*m)*h*(E_NA - v) + G_KD*(n*n*n*n)*(E_K - v) + G_L*(E_L - v)
          + ge*(E_EXC - v) + gi*(E_INH - v)
          + I_drive + I_THETA*sin(2*pi*F_THETA*t) ) / C_M : volt
dm/dt = 0.32*(mV**-1)*(13.*mV-v+V_T)/
        (exp((13.*mV-v+V_T)/(4.*mV))-1.)/ms*(1-m)
       -0.28*(mV**-1)*(v-V_T-40.*mV)/
        (exp((v-V_T-40.*mV)/(5.*mV))-1.)/ms*m : 1
dn/dt = 0.032*(mV**-1)*(15.*mV-v+V_T)/
        (exp((15.*mV-v+V_T)/(5.*mV))-1.)/ms*(1.-n)
       -.5*exp((10.*mV-v+V_T)/(40.*mV))/ms*n : 1
dh/dt = 0.128*exp((17.*mV-v+V_T)/(18.*mV))/ms*(1.-h)
       -4./(1+exp((40.*mV-v+V_T)/(5.*mV)))/ms*h : 1
dge/dt = -ge/TAU_AMPA : siemens
dgi/dt = -gi/tau_gaba : siemens
I_drive : amp
"""

NS = dict(G_L=G_L, E_L=E_L, E_EXC=E_EXC, E_INH=E_INH, C_M=C_M, TAU_M=TAU_M,
          TAU_AMPA=TAU_AMPA, I_THETA=I_THETA, F_THETA=F_THETA, SIGMA_V=SIGMA_V,
          V_TH=V_TH, V_RESET=V_RESET, G_NA=G_NA, G_KD=G_KD, E_NA=E_NA, E_K=E_K,
          V_T=V_T)


# ============================================================================
# THE TWO LINES v1 NEEDED
# ============================================================================
def assert_connected(syns, labels, n_targets):
    """Every declared pathway must actually contain synapses."""
    print("    connectivity self-check:")
    for S, lab, ntgt in zip(syns, labels, n_targets):
        deg = len(S) / max(ntgt, 1)
        print(f"      {lab}: {len(S):6d} synapses | mean in-degree {deg:6.2f}")
        if len(S) == 0:
            raise AssertionError(
                f"{lab} has ZERO synapses. This is the v1 failure. Refusing to run.")


def assert_firing(rates, labels):
    """Every population must actually spike. (v1's HH interneurons never did.)"""
    print("    firing self-check:")
    for r, lab in zip(rates, labels):
        print(f"      {lab}: {r:6.2f} Hz")
        if r <= 0.0:
            raise AssertionError(
                f"{lab} never fired. Check its drive vs rheobase. "
                f"This is the v1 hybrid-network failure. Refusing to run.")


# ============================================================================
# NETWORK
# ============================================================================
def build_and_run(n_exc, n_inh, int_model="lif", tau_gaba=TAU_GABA_DEFAULT,
                  p_ie=P_IE_BASE, p_ee=P_EE_BASE, p_ei=P_EI_BASE, p_ii=P_II_BASE,
                  seed_val=100, verbose=False, normalize=True):
    """normalize=True  -> weights rescaled so total synaptic drive per cell is held
                          constant. Use for the SCALE arm: then size is the only
                          thing that varies.
       normalize=False -> raw weights. Use for the UNCOMPENSATED sparsification
                          arm: then cutting p_ie really does cut total inhibition.
       Sparsification is run BOTH ways below, because "fewer connections" and
       "fewer connections at matched drive" are different questions and silently
       picking one is how v1 got into trouble."""
    start_scope()
    brian_seed(seed_val)
    np.random.seed(seed_val)
    defaultclock.dt = DT
    ns = dict(NS, tau_gaba=tau_gaba)

    exc = NeuronGroup(n_exc, EQS_LIF, threshold="v > V_TH", reset="v = V_RESET",
                      refractory=REFRAC_PYR, method="euler", namespace=ns)
    exc.v, exc.I_drive = E_L, I_DRIVE_PYR

    if int_model == "lif":
        inh = NeuronGroup(n_inh, EQS_LIF, threshold="v > V_TH", reset="v = V_RESET",
                          refractory=REFRAC_INT, method="euler", namespace=ns)
        inh.v, inh.I_drive = E_L, I_DRIVE_INT_LIF        # <- LIF drive to LIF cell
    elif int_model == "hh":
        inh = NeuronGroup(n_inh, EQS_HH, threshold="v > -20*mV",
                          refractory="v > -40*mV", method="exponential_euler",
                          namespace=ns)
        inh.v, inh.m, inh.n, inh.h = E_L, 0.05, 0.32, 0.6
        inh.I_drive = I_DRIVE_INT_HH                      # <- HH drive to HH cell
    else:
        raise ValueError(int_model)

    if normalize:
        w_ee, w_ei, w_ie, w_ii = scaled_weights(n_exc, n_inh, p_ee, p_ei, p_ie, p_ii)
    else:
        w_ee, w_ei, w_ie, w_ii = W_EE_REF, W_EI_REF, W_IE_REF, W_II_REF
    S_ee = Synapses(exc, exc, on_pre="ge_post += W", namespace={"W": w_ee})
    S_ei = Synapses(exc, inh, on_pre="ge_post += W", namespace={"W": w_ei})
    S_ie = Synapses(inh, exc, on_pre="gi_post += W", namespace={"W": w_ie})
    S_ii = Synapses(inh, inh, on_pre="gi_post += W", namespace={"W": w_ii})
    S_ee.connect(p=p_ee, condition="i != j")
    S_ei.connect(p=p_ei)
    S_ie.connect(p=p_ie)
    S_ii.connect(p=p_ii, condition="i != j")

    if verbose:
        assert_connected([S_ee, S_ei, S_ie, S_ii],
                         ["E->E", "E->I", "I->E", "I->I"],
                         [n_exc, n_inh, n_exc, n_inh])
    else:
        for S, lab in zip([S_ee, S_ei, S_ie, S_ii], ["E->E", "E->I", "I->E", "I->I"]):
            if len(S) == 0:
                raise AssertionError(f"{lab} has ZERO synapses (v1 failure).")

    spk_e, spk_i = SpikeMonitor(exc), SpikeMonitor(inh)
    net = Network(exc, inh, S_ee, S_ei, S_ie, S_ii, spk_e, spk_i)

    t0 = time.time()
    net.run(DURATION)
    cost = time.time() - t0

    T = float((DURATION - TRANSIENT) / second)
    t_e = np.asarray(spk_e.t / second); t_e = t_e[t_e > float(TRANSIENT / second)]
    t_i = np.asarray(spk_i.t / second); t_i = t_i[t_i > float(TRANSIENT / second)]
    r_e, r_i = len(t_e) / (n_exc * T), len(t_i) / (n_inh * T)

    if verbose:
        assert_firing([r_e, r_i], ["pyramidal", "interneuron"])
    elif r_i <= 0:
        raise AssertionError("interneurons never fired (v1 hybrid failure).")

    return dict(spikes_e=t_e - float(TRANSIENT / second), n_exc=n_exc, T=T,
                rate_e=r_e, rate_i=r_i, cost_s=cost,
                n_ie=len(S_ie), n_ei=len(S_ei))


def rhythm_pvals(res, seed=0):
    """NOTE: drive_hz MUST be passed. Without it only the surrogate null runs,
    and the null alone reports the 9th harmonic of the 6 Hz drive (54.9 Hz) as
    'gamma' -- which is the entire point of this paper. The verdict below uses
    is_rhythm (all four screens), never `significant` (the null alone)."""
    g = test_rhythm(res["spikes_e"], res["n_exc"], res["T"], band=GAMMA_BAND,
                    n_surr=N_SURR, method="jitter", seed=seed,
                    drive_hz=DRIVE_HZ)
    # theta IS the drive, so drive-harmonic screening is meaningless for it
    # (k=1 would reject the thing we are trying to detect). Rate-harmonic
    # screening is also off: the firing rate is entrained BY theta.
    t = test_rhythm(res["spikes_e"], res["n_exc"], res["T"], band=THETA_BAND,
                    n_surr=N_SURR, method="jitter", seed=seed,
                    drive_hz=None, check_rate_harmonic=False)
    return g, t


# ============================================================================
# THE BATTERY  (v1's arms, now with a circuit that exists)
# ============================================================================
CONDITIONS = [
    # name,                          n_exc, n_inh, int_model, p_ie,          normalize
    ("LIF baseline",                    80,   20,   "lif",  P_IE_BASE,        True),
    ("Sparsified -30% (uncompensated)", 80,   20,   "lif",  P_IE_BASE * 0.7,  False),
    ("Sparsified -30% (drive-matched)", 80,   20,   "lif",  P_IE_BASE * 0.7,  True),
    ("Hybrid (HH interneurons)",        80,   20,   "hh",   P_IE_BASE,        True),
    ("Scaled 4x (drive-matched)",      320,   80,   "lif",  P_IE_BASE,        True),
]


def calibrate_w_ie(seeds=(100, 101, 102)):
    """THE POSITIVE CONTROL, RUN RATHER THAN REMEMBERED.

    W_IE = 6 nS is not a guess: it was calibrated by requiring the pipeline to
    recover the 6 Hz theta that we KNOW is injected. A test that cannot see a
    rhythm placed there by hand has no standing to say anything about one that
    might not be there.

    This sweep used to live in a COMMENT above W_IE_REF. The numbers were real,
    but a number in a comment is a number no one can check, and running this file
    did not produce them. Section 2.5 of the manuscript quoted them. It is a paper
    about archives that fail to produce what they claim. So it runs now.

    The criterion never examines the gamma band, so it cannot bias the gamma result."""
    print("=" * 92)
    print("POSITIVE CONTROL — calibration of W_IE by theta recovery")
    print("=" * 92)
    print("  Sweep the inhibitory weight. Keep the range where the pipeline still")
    print("  recovers the 6 Hz theta that IS IN THE INPUT. This never looks at gamma.\n")
    print(f"  {'W_IE':>7} {'rate_E':>9} {'theta p':>10} {'theta recovered in':>20}")
    print("  " + "-" * 52)
    global W_IE_REF
    keep = W_IE_REF
    out = []
    try:
        # try/finally, not a bare assignment at the end. build_and_run() raises
        # AssertionError on a disconnected pathway or a silent population. If it
        # raises on the last sweep value, a bare restore never executes and
        # W_IE_REF is left at 20 nS for the rest of the process -- every
        # subsequent scaled_weights() call silently uses the wrong inhibition,
        # with nothing in the output saying so.
        for w in (3.0, 6.0, 10.0, 20.0):
            W_IE_REF = w * nS
            rates, ps, rec = [], [], []
            for s in seeds:
                res = build_and_run(N_EXC_REF, N_INH_REF, seed_val=s)
                _, t = rhythm_pvals(res, seed=s)
                rates.append(res["rate_e"]); ps.append(t["p"])
                rec.append(t["significant"])
            adopted = abs(w - W_IE_ADOPTED_NS) < 1e-9
            tag = " (adopted)" if adopted else ""
            print(f"  {w:>4.0f} nS {np.mean(rates):>8.1f} Hz {np.median(ps):>10.3f} "
                  f"{100*np.mean(rec):>17.0f}%{tag}")
            out.append((w, float(np.mean(rates)), float(np.median(ps)),
                        float(np.mean(rec))))
    finally:
        W_IE_REF = keep

    # STATE WHAT THE SWEEP ACTUALLY SHOWED, not what it showed a month ago.
    #
    # This function used to print, unconditionally and regardless of the numbers
    # immediately above it: "At 20 nS the pipeline fails to detect a rhythm that
    # is unambiguously present." That was measured at a 3 s recording length. At
    # the 21 s length that section 2.3 derives, it is false -- theta is recovered
    # at every weight. The sentence was a conclusion hard-coded into a print
    # statement, in the positive control of a paper about conclusions that outlive
    # the measurements behind them.
    n_ok = sum(1 for _, _, _, r in out if r == 1.0)
    print()
    print(f"  Theta is recovered in 100% of seeds at {n_ok} of {len(out)} weights.")
    if n_ok == len(out):
        print("  => The control PASSES: the pipeline can see a rhythm that is known to")
        print("     be present, at every weight tested. It therefore has the power to")
        print("     say something about a rhythm that might not be.")
        print()
        print("     It does NOT discriminate among these weights, and no claim that it")
        print("     does should be made. W_IE = 6 nS is adopted because it yields a")
        print("     pyramidal rate of ~6 Hz, which is the physiological target. The")
        print("     choice is made on the rate column, not on the theta column.")
    else:
        print("  *** The pipeline is BLIND at some weights. It cannot recover a rhythm")
        print("  *** placed there by hand. Nothing downstream may be reported.")
        raise SystemExit("positive control failed -- do not report these results")
    print()
    return out


def main():
    df = (1.0 / DT_S) / NPERSEG
    print_version_banner({
        "rate binning": f"{DT_S*1000:.0f} ms",
        "nperseg": NPERSEG,
        "freq resolution df": f"{df:.3f} Hz",
        "recording": f"{DURATION/second:.0f} s (minus {TRANSIENT/second:.0f} s transient)",
        "comb notch": "ON (drive harmonics excised before max)",
        "harmonic screen blocks": f"{100*blocked_fraction(df, DRIVE_HZ):.0f}% of the gamma band",
        "N_surr": N_SURR,
    })
    print()
    print("=" * 92)
    print("CA1 v2 — the corrected study")
    print("=" * 92)
    print(f"per-type connectivity: p_ee={P_EE_BASE} p_ei={P_EI_BASE} "
          f"p_ie={P_IE_BASE} p_ii={P_II_BASE}")
    print(f"drives (separately named): pyr={I_DRIVE_PYR} | int_LIF={I_DRIVE_INT_LIF} "
          f"| int_HH={I_DRIVE_INT_HH}")
    print(f"surrogates per test: {N_SURR} (jitter) | network seeds: {SEEDS}\n")

    calibrate_w_ie()

    print("-" * 92)
    print("PRE-FLIGHT (the two assertions v1 lacked)")
    print("-" * 92)
    for name, ne, ni, im, pie, nrm in CONDITIONS:
        print(f"  {name}")
        build_and_run(ne, ni, int_model=im, p_ie=pie, seed_val=SEEDS[0],
                      verbose=True, normalize=nrm)
        print()

    print("-" * 92)
    print("RESULTS — rhythms tested against a surrogate null, NOT raw dB")
    print("-" * 92)
    hdr = (f"{'condition':<28}{'rate_E':>7}{'rate_I':>7}{'cost(s)':>8}"
           f"{'theta p':>9}{'gamma p':>9}{'gamma dB':>9}{'null dB':>8}  verdict")
    print(hdr); print("-" * len(hdr))

    rows = []
    for name, ne, ni, im, pie, nrm in CONDITIONS:
        gp, tp, gd, nd, gf, re_, ri_, cs = [], [], [], [], [], [], [], []
        g_null, g_rhy, t_rhy, why = [], [], [], []
        for s in SEEDS:
            r = build_and_run(ne, ni, int_model=im, p_ie=pie, seed_val=s,
                              normalize=nrm)
            g, t = rhythm_pvals(r, seed=s)
            gp.append(g["p"]); tp.append(t["p"])
            gd.append(g["obs_db"]); nd.append(g["null_mean"]); gf.append(g["fpk"])
            g_null.append(g["significant"])     # the NULL alone
            g_rhy.append(g["is_rhythm"])        # ALL FOUR screens
            t_rhy.append(t["is_rhythm"])
            why.append(g["reason"])
            re_.append(r["rate_e"]); ri_.append(r["rate_i"]); cs.append(r["cost_s"])

        null_says = np.mean(g_null) > 0.5      # what the null alone concludes
        screens_say = np.mean(g_rhy) > 0.5     # what all four screens conclude
        theta_ok = np.mean(t_rhy) > 0.5
        v = ("GAMMA" if screens_say else "no gamma") + \
            (" | THETA" if theta_ok else " | no theta")
        flag = "  <-- NULL ALONE SAYS GAMMA" if (null_says and not screens_say) else ""
        print(f"{name:<28}{np.mean(re_):>7.1f}{np.mean(ri_):>7.1f}{np.mean(cs):>8.1f}"
              f"{np.median(tp):>9.3f}{np.median(gp):>9.3f}"
              f"{np.mean(gd):>9.2f}{np.mean(nd):>8.2f}  {v}{flag}", flush=True)
        if null_says and not screens_say:
            # NEVER average peak frequencies across seeds: the peak migrates between
            # adjacent teeth of the harmonic comb, and the mean of such values can
            # land on no harmonic at all. Report them individually.
            print(f"{'':<28}  the null alone reports gamma. Per-seed peaks:", flush=True)
            for s_, f_, w_ in zip(SEEDS, gf, why):
                k_ = round(f_ / DRIVE_HZ)
                print(f"{'':<30}seed {s_}: {f_:5.1f} Hz -> {k_}x{DRIVE_HZ:.0f}"
                      f"={k_*DRIVE_HZ:.0f} Hz | {w_}", flush=True)
        rows.append((name, np.median(tp), np.median(gp), screens_say, theta_ok,
                     np.mean(gd), np.mean(nd), null_says, np.mean(gf)))

    print()
    print("=" * 92)
    print("READ THIS BEFORE WRITING ANYTHING")
    print("=" * 92)
    print(f"  The null of the prominence statistic is ~{np.mean([r[6] for r in rows]):.1f} dB.")
    print("  Any 'prominence' at or below that number is NOT evidence of a rhythm —")
    print("  it is the statistic measuring its own bias. v1 reported theta at 6.6-7.5 dB.")
    print()
    fooled = [r for r in rows if r[7] and not r[3]]
    if fooled:
        print("  The surrogate null ALONE reported gamma in "
              f"{len(fooled)}/{len(rows)} conditions:")
        for r in fooled:
            print(f"      {r[0]}: median p={r[2]:.3f} (see per-seed peaks above)")
        df = (1.0 / DT_S) / NPERSEG
        print("  All were rejected by a harmonic screen: the surrogate null alone is")
        print("  not sufficient, which is this paper's central claim, reproduced here")
        print("  inside its own pipeline.")
        print()
        print(f"  CAVEAT. At df = {df:.3f} Hz the screen rejects "
              f"{100*blocked_fraction(df, F_THETA/Hz):.0f}% of the 30-80 Hz band. It")
        print("  shows a peak is NOT DISTINGUISHABLE from a drive harmonic; it does not")
        print("  prove that it IS one. Only the tau_GABA scaling test separates them.")
        print()
    if not any(r[3] for r in rows):
        print("  No condition shows emergent gamma against the null.")
        print("  Unlike v1, this is now an INTERPRETABLE negative result: the E-I loop")
        print("  exists (I->E in-degree > 0), the interneurons fire, and there is still")
        print("  no gamma. The absence is a property of the parameter regime, not of a")
        print("  disconnected network.")
    else:
        print("  Gamma survives the null in at least one condition. Report the p, not the dB,")
        print("  and check that its peak frequency scales with tau_GABA before calling it PING.")
    print("=" * 92)


if __name__ == "__main__":
    main()
