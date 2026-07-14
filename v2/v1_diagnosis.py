#!/usr/bin/env python3
"""
v1_diagnosis.py -- the three measurements the corrigendum rests on.
Larry (Peilin) Zhong

WHY THIS FILE EXISTS
--------------------
The corrigendum makes three quantitative claims about Version 1:

  C2  the hybrid interneurons were driven at I = 1.0 uA/cm^2, and that HH
      neuron's rheobase is between 6 and 7, so they emitted zero spikes;
  M1  the generator that injected gamma by PHASE-MODULATING spike times does
      not produce a clean rhythm, it produces intermodulation sidebands, so a
      nominal 40 Hz injection peaks somewhere else entirely;
  M2  the harmonic screen blocks a fraction of the analysis band that depends
      on the frequency resolution, and at v1's resolution it blocked so much
      of the band that it would have destroyed a real rhythm.

All three numbers appeared in the manuscript. NONE of them was produced by any
script in this archive: they were measured during development, by code that was
never committed. The numbers were real. That is not sufficient, and in this
paper of all papers it is not sufficient, because the entire argument is that a
number whose provenance cannot be checked is a number a reader cannot use.

So they are measured here. This file simulates no network and takes seconds.

Run:  python3 v1_diagnosis.py
"""

import numpy as np
from brian2 import (start_scope, NeuronGroup, SpikeMonitor, run, defaultclock,
                    ms, second, uA, cm, mV, ufarad, msiemens)
from spectral_null import (pop_rate, psd, prominence, blocked_fraction,
                           required_df, print_version_banner, DT_S)


# ============================================================================
# C1. v1's networks contained essentially no synapses.
# ============================================================================
def c1_synapse_counts():
    """Re-draw v1's connectivity, from v1's own line, with v1's own seeds.

    v1/CA1-simplification-v3.py:100 draws every pathway from ONE probability:

        conn = (rng.random((N, N)) < p_conn)      # p_conn = 0.02

    2% is an empirical fact about pyramidal-to-pyramidal connectivity in CA1,
    which is famously sparse. It is not the local connection probability of
    interneurons onto pyramidal cells, which is an order of magnitude higher.
    Applied to a 10-neuron network the expected number of inhibitory-to-
    excitatory synapses is 2 * 8 * 0.02 = 0.32.

    This is the corrigendum's first claim, and it was the last one still resting
    on a table with no script behind it."""
    print("=" * 80)
    print("C1  v1's REALISED CONNECTIVITY, redrawn from v1's own source and seeds")
    print("=" * 80)
    N_EXC, N_INH, P = 8, 2, 0.02
    N = N_EXC + N_INH
    print(f"  N_exc = {N_EXC}, N_inh = {N_INH}, one p_conn = {P} for EVERY pathway.")
    print(f"  Expected I->E synapses: {N_INH} x {N_EXC} x {P} = {N_INH*N_EXC*P:.2f}\n")
    print(f"  {'seed':>6} {'I->E synapses':>15} {'TOTAL synapses':>16}")
    print("  " + "-" * 40)
    ie_all, tot_all = [], []
    for seed in (100, 101, 102, 103, 104):
        rng = np.random.default_rng(seed)
        conn = rng.random((N, N)) < P
        ie = int(conn[N_EXC:, :N_EXC].sum())
        tot = int(conn.sum())
        ie_all.append(ie)
        tot_all.append(tot)
        print(f"  {seed:>6} {ie:>15} {tot:>16}")
    print()
    print(f"  I->E across seeds : {ie_all}")
    print(f"  TOTAL across seeds: {tot_all}")
    print()
    dead = sum(t == 0 for t in tot_all)
    print(f"  => Not one inhibitory-to-excitatory synapse exists on ANY seed, and")
    print(f"     {dead} of 5 seeds produced a network with zero synapses of any kind.")
    print(f"     v1's 'microcircuit' was ten uncoupled neurons sharing a 6 Hz drive.")
    print(f"     The sparsification arm removed connections from a network that had")
    print(f"     none. The absence of emergent gamma was uninformative, in the")
    print(f"     DETAILED configuration exactly as much as in the reduced ones.")
    print()
    if any(ie_all):
        print("  *** C1 IS NOT REPRODUCED. Do not report it as written. ***")
        raise SystemExit(1)
    return ie_all, tot_all


# ============================================================================
# C2. The rheobase of the Hodgkin-Huxley interneuron v1 used.
# ============================================================================
# v1 passed I_mean_lif = 1.0 to this neuron. The single-neuron HH block in the
# same file used I = 10.0. Nothing in v1 checked that the hybrid interneurons
# ever fired.
#
# v1 used the CLASSICAL Hodgkin-Huxley squid-axon model (g_Na = 120, g_K = 36,
# g_L = 0.3, E_L = -54.4), whose rheobase is ~6 uA/cm^2. Not the Wang-Buzsaki
# interneuron (g_Na = 35, g_K = 9, g_L = 0.1, E_L = -65) that this paper's own
# network uses, whose rheobase is far below 1. The constants below are transcribed
# from v1/CA1-simplification-v3.py, line 50.
HH_EQS = """
dv/dt = (I - g_na*m**3*h*(v - E_na) - g_k*n**4*(v - E_k)
         - g_l*(v - E_l)) / C : volt
dm/dt = alpha_m*(1-m) - beta_m*m : 1
dh/dt = alpha_h*(1-h) - beta_h*h : 1
dn/dt = alpha_n*(1-n) - beta_n*n : 1
alpha_m = (0.1/mV)*(v+40*mV)/(1-exp(-(v+40*mV)/(10*mV)))/ms : Hz
beta_m  = 4*exp(-(v+65*mV)/(18*mV))/ms : Hz
alpha_h = 0.07*exp(-(v+65*mV)/(20*mV))/ms : Hz
beta_h  = 1./(exp(-(v+35*mV)/(10*mV))+1)/ms : Hz
alpha_n = (0.01/mV)*(v+55*mV)/(1-exp(-(v+55*mV)/(10*mV)))/ms : Hz
beta_n  = 0.125*exp(-(v+65*mV)/(80*mV))/ms : Hz
I : amp/meter**2
"""

# transcribed from v1/CA1-simplification-v3.py:50
V1_HH = dict(C=1 * ufarad / cm ** 2,
             g_na=120 * msiemens / cm ** 2, E_na=50 * mV,
             g_k=36 * msiemens / cm ** 2, E_k=-77 * mV,
             g_l=0.3 * msiemens / cm ** 2, E_l=-54.4 * mV)


SETTLE = 200 * ms     # onset transient, discarded


def hh_spike_count(I_val, T=1000 * ms, dt=0.01 * ms):
    """Returns (total spikes, SUSTAINED spikes after the onset transient).

    The distinction is not pedantry. Starting the membrane at -65 mV when the
    leak reversal is -54.4 mV depolarises the cell, and it fires ONE spike on
    the way -- at every current, including currents far below rheobase. Counting
    that spike makes a silent neuron look like a firing one. Rheobase is a
    property of the SUSTAINED response, so the first 200 ms is discarded."""
    start_scope()
    defaultclock.dt = dt
    G = NeuronGroup(1, HH_EQS, threshold="v > 0*mV", refractory="v > 0*mV",
                    method="exponential_euler", namespace=V1_HH)
    G.v = -65 * mV
    G.h = 1.0
    G.I = I_val * uA / cm ** 2
    M = SpikeMonitor(G)
    run(T)
    t = M.t
    return int(M.num_spikes), int(np.sum(t > SETTLE))


def c2_rheobase():
    print("=" * 80)
    print("C2  RHEOBASE OF THE HH INTERNEURON  (v1 drove it at I = 1.0)")
    print("=" * 80)
    print("  1000 ms per current. 'sustained' discards the first 200 ms, in which")
    print("  the cell fires one onset spike at EVERY current as it depolarises from")
    print("  -65 mV to the -54.4 mV leak reversal.\n")
    print(f"  {'I (uA/cm^2)':>12} {'total':>7} {'sustained':>10}   note")
    print("  " + "-" * 60)
    sus = {}
    for I in (1.0, 3.0, 5.0, 6.0, 7.0, 10.0):
        tot, s = hh_spike_count(I)
        sus[I] = s
        note = ""
        if I == 1.0:
            note = "<- v1 drove the HYBRID interneurons here"
        elif I == 10.0:
            note = "<- v1's own single-neuron block used this"
        print(f"  {I:>12.1f} {tot:>7} {s:>10}   {note}")
    silent = [i for i, n in sus.items() if n == 0]
    firing = [i for i, n in sus.items() if n > 0]
    print()
    if not (silent and firing):
        print("  *** C2 IS NOT REPRODUCED. Do not report it as written. ***")
        raise SystemExit(1)
    lo, hi = max(silent), min(firing)
    print(f"  => Rheobase is between {lo:.0f} and {hi:.0f} uA/cm^2. At I = 1.0 the neuron")
    print(f"     emits {sus[1.0]} sustained spikes. The hybrid arm of v1 -- whose stated")
    print(f"     purpose was to test whether detailed inhibitory kinetics recover")
    print(f"     gamma -- contained interneurons that did not fire.")
    print()
    print(f"     v1 integrated this model with a hand-written forward Euler step;")
    print(f"     this is Brian2's exponential Euler at dt = 0.01 ms. Spike COUNTS")
    print(f"     above rheobase therefore differ by a spike or two between the two.")
    print(f"     The rheobase, and the silence at I = 1.0, do not.")
    print()
    return sus


# ============================================================================
# M1. The discarded gamma generator produced sidebands, not a rhythm.
# ============================================================================
def m1_phase_modulation_sidebands():
    print("=" * 80)
    print("M1  WHY THE PHASE-MODULATION GAMMA GENERATOR WAS DISCARDED")
    print("=" * 80)
    print("  Injecting gamma by displacing spike times with a sinusoid does not add")
    print("  a rhythm. It MULTIPLIES the 6 Hz comb by the injected frequency, and")
    print("  what appears is the intermodulation product, not the injection.\n")
    T, N, DRIVE = 21.0, 80, 6.0
    print(f"  {'nominal injection':>18} {'peak actually found':>21}")
    print("  " + "-" * 44)
    for f_inject in (40.0, 55.0):
        rng = np.random.default_rng(9000)
        n_cyc = int(T * DRIVE)
        per = max(int(round(6.0 * N / DRIVE)), 1)
        st = np.concatenate([((c + 0.25) / DRIVE)
                             + rng.normal(0, 0.005, per) for c in range(n_cyc)])
        st = st + (0.5 / (2 * np.pi * f_inject)) * np.sin(2 * np.pi * f_inject * st)
        st = np.sort(st[(st >= 0) & (st < T)])
        f, P = psd(pop_rate(st, N, T), 1.0 / DT_S)
        _, fpk, _ = prominence(f, P, (f_inject - 8, f_inject + 20))
        flag = "" if abs(fpk - f_inject) < 1.0 else "  <- NOT the injected frequency"
        print(f"  {f_inject:>15.0f} Hz {fpk:>18.1f} Hz{flag}")
    print()
    print("  A positive control that does not contain the rhythm it claims to")
    print("  contain is worthless. It was replaced by ground_truth(), which is the")
    print("  UNION of two INDEPENDENT volley trains and therefore contains exactly")
    print("  the frequency it says it does.")
    print()


# ============================================================================
# M2. How much of the band the harmonic screen destroys, vs resolution.
# ============================================================================
def m2_screen_cost_vs_resolution():
    print("=" * 80)
    print("M2  THE HARMONIC SCREEN'S COST DEPENDS ON THE FREQUENCY RESOLUTION")
    print("=" * 80)
    print("  The screen rejects a tolerance of +/-1.5 bins around every multiple of")
    print("  the drive, so it destroys ~3*df/f_drive of the band. At coarse")
    print("  resolution it destroys most of it -- INCLUDING any real rhythm.\n")
    print(f"  {'T (s)':>7} {'nperseg':>9} {'df (Hz)':>9} {'band blocked':>14}   verdict")
    print("  " + "-" * 62)
    fs = 1.0 / DT_S
    for T in (2.5, 3.0, 5.0, 10.0, 21.0):
        nps = int(min(T / DT_S, 8192))
        df = fs / nps
        frac = blocked_fraction(df, 6.0)
        ok = "usable" if frac <= 0.10 else "*** DESTROYS THE BAND ***"
        print(f"  {T:>7.1f} {nps:>9} {df:>9.3f} {100*frac:>13.0f}%   {ok}")
    need = required_df(0.10, 6.0)
    print()
    print(f"  => Requiring the blocked fraction to be at most 10% gives")
    print(f"     df <= 0.1 * f_drive / 3 = {need:.2f} Hz, i.e. a Welch segment of")
    print(f"     at least {int(np.ceil(fs/need))} samples = {fs/need/1000:.1f} s. All results here use 21 s.")
    print()


if __name__ == "__main__":
    print_version_banner()
    print()
    c1_synapse_counts()
    c2_rheobase()
    m1_phase_modulation_sidebands()
    m2_screen_cost_vs_resolution()
    print("=" * 80)
    print("Every number in the corrigendum, and the two Methods numbers that")
    print("justified discarding v1's positive control and v1's recording length,")
    print("is produced above by code in this archive. Until this file existed they")
    print("were carried in prose, and prose does not recompute itself when the code")
    print("underneath it changes. That is not a stylistic complaint. It is the")
    print("finding of this paper, applied to this paper.")
    print("=" * 80)
