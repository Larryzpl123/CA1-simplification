# CA1 microcircuit simplification study - revised analysis (v3)
# Larry (Peilin) Zhong, https://larryzpl123.github.io/
#
# Key corrections vs. the original script:
#   1. HH single-neuron drive raised so the cell fires throughout the window
#      (the old I_mean produced only a startup burst, then silence).
#   2. Fair timing: HH and LIF single-neuron models are BOTH integrated with the
#      same fixed-step forward-Euler scheme at the same dt, so compute-time
#      differences reflect model complexity (4 ODEs vs 1), not the solver.
#   3. Honest spectral metrics: instead of "max raw power in band" (dominated by
#      the 1/f background), we report (a) relative band power and (b) peak
#      prominence above a fitted 1/f aperiodic background, which actually tests
#      whether a rhythm exists.
#   4. Stochastic input (seeded per run) so the reported SDs are meaningful.
#   5. Per-block checkpointing so the full battery can be completed across
#      several invocations if needed.

import numpy as np
from scipy.signal import find_peaks, welch
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time, os, pickle, sys

OUTDIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(OUTDIR, "checkpoints_v2")
os.makedirs(CKPT, exist_ok=True)

# ----------------------------------------------------------------------------
# Hodgkin-Huxley gating kinetics
# ----------------------------------------------------------------------------
def alpha_m(V):
    den = 1 - np.exp(-(V + 40) / 10)
    return np.where(np.abs(den) < 1e-8, 1.0, 0.1 * (V + 40) / den)
def beta_m(V):  return 4 * np.exp(-(V + 65) / 18)
def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
def beta_h(V):  return 1 / (1 + np.exp(-(V + 35) / 10))
def alpha_n(V):
    den = 1 - np.exp(-(V + 55) / 10)
    return np.where(np.abs(den) < 1e-8, 0.1, 0.01 * (V + 55) / den)
def beta_n(V):  return 0.125 * np.exp(-(V + 65) / 80)

THETA_HZ = 6.0

# ----------------------------------------------------------------------------
# Single HH neuron, fixed-step forward Euler
# ----------------------------------------------------------------------------
def hh_single_euler(t_span, dt, I_mean, theta_amp=2.0, noise_sigma=0.0, seed=0,
                    g_Na=120, g_K=36, g_L=0.3, E_Na=50, E_K=-77, E_L=-54.4, C_m=1):
    rng = np.random.default_rng(seed)
    t = np.arange(t_span[0], t_span[1], dt)
    n_steps = len(t)
    V = np.empty(n_steps); V[0] = -65.0
    m, h, n = 0.05, 0.6, 0.32
    drive = I_mean + theta_amp * np.sin(2*np.pi*THETA_HZ*t/1000.0)
    if noise_sigma > 0:
        drive = drive + noise_sigma * rng.standard_normal(n_steps)
    for i in range(1, n_steps):
        Vp = V[i-1]
        dV = (-g_Na*m**3*h*(Vp-E_Na) - g_K*n**4*(Vp-E_K) - g_L*(Vp-E_L) + drive[i]) / C_m
        m += dt * (alpha_m(Vp)*(1-m) - beta_m(Vp)*m)
        h += dt * (alpha_h(Vp)*(1-h) - beta_h(Vp)*h)
        n += dt * (alpha_n(Vp)*(1-n) - beta_n(Vp)*n)
        V[i] = Vp + dt * dV
    return t, V

def lif_single_euler(t_span, dt, I_mean, theta_amp=0.2, noise_sigma=0.0, seed=0,
                     tau=20, E_L=-65, R=10, V_th=-55, V_reset=-65):
    rng = np.random.default_rng(seed)
    t = np.arange(t_span[0], t_span[1], dt)
    n_steps = len(t)
    V = np.empty(n_steps); V[0] = E_L
    drive = I_mean + theta_amp * np.sin(2*np.pi*THETA_HZ*t/1000.0)
    if noise_sigma > 0:
        drive = drive + noise_sigma * rng.standard_normal(n_steps)
    spikes = []
    for i in range(1, n_steps):
        v = V[i-1] + dt * (-(V[i-1] - E_L) + R * drive[i]) / tau
        if v > V_th:
            v = V_reset; spikes.append(t[i])
        V[i] = v
    return t, V, spikes

# ----------------------------------------------------------------------------
# LIF / hybrid networks, fixed-step Euler (optimized)
# ----------------------------------------------------------------------------
def _network(t_span, dt, N_exc, N_inh, p_conn, I_mean, hybrid,
             theta_amp=0.2, noise_sigma=0.3, seed=42,
             tau=20, E_L=-65, R=10, V_th=-55, V_reset=-65,
             tau_syn=5, E_syn_exc=0, E_syn_inh=-80):
    rng = np.random.default_rng(seed)
    base_g_exc = 0.05 * 8
    base_g_inh = 0.1 * 2
    N = N_exc + N_inh
    t = np.arange(t_span[0], t_span[1], dt)
    n_steps = len(t)
    V = np.empty((n_steps, N)); V[0] = E_L
    s = np.zeros((N, N))
    conn = (rng.random((N, N)) < p_conn).astype(float)
    w_exc = base_g_exc * conn[:N_exc, :] / N_exc
    w_inh = base_g_inh * conn[N_exc:, :] / N_inh
    theta = I_mean + theta_amp * np.sin(2*np.pi*THETA_HZ*t/1000.0)
    noise = noise_sigma * rng.standard_normal((n_steps, N))
    spike_mat = np.zeros((n_steps, N), dtype=bool)
    if hybrid:
        m_inh = np.full(N_inh, 0.05); h_inh = np.full(N_inh, 0.6); n_inh = np.full(N_inh, 0.32)
    decay = dt / tau_syn
    for i in range(1, n_steps):
        I_ext = theta[i] + noise[i]
        gs_exc = np.sum(w_exc * s[:N_exc, :], axis=0)
        gs_inh = np.sum(w_inh * s[N_exc:, :], axis=0)
        Vp = V[i-1]
        I_syn = gs_exc * (E_syn_exc - Vp) + gs_inh * (E_syn_inh - Vp)
        if hybrid:
            Vn = Vp.copy()
            # exc: LIF
            Vn[:N_exc] = Vp[:N_exc] + dt * (-(Vp[:N_exc] - E_L) + R*I_ext[:N_exc] + R*I_syn[:N_exc]) / tau
            # inh: HH
            Vi = Vp[N_exc:]
            curr = (-120*m_inh**3*h_inh*(Vi-50) - 36*n_inh**4*(Vi+77) - 0.3*(Vi+54.4)
                    + I_ext[N_exc:] + I_syn[N_exc:])
            Vn[N_exc:] = Vi + dt * curr
            m_inh += dt * (alpha_m(Vi)*(1-m_inh) - beta_m(Vi)*m_inh)
            h_inh += dt * (alpha_h(Vi)*(1-h_inh) - beta_h(Vi)*h_inh)
            n_inh += dt * (alpha_n(Vi)*(1-n_inh) - beta_n(Vi)*n_inh)
            # LIF pyramidals: threshold + reset. HH interneurons: NO manual
            # reset (the HH dynamics repolarize on their own); a spike is a
            # genuine action potential, detected as an upward crossing of 0 mV.
            spiked_exc = Vn[:N_exc] > V_th
            Vn[:N_exc][spiked_exc] = V_reset
            spiked_inh = (Vi < 0) & (Vn[N_exc:] >= 0)
            spiked = np.concatenate((spiked_exc, spiked_inh))
        else:
            Vn = Vp + dt * (-(Vp - E_L) + R * I_ext + R * I_syn) / tau
            spiked = Vn > V_th
            Vn[spiked] = V_reset
        V[i] = Vn
        spike_mat[i] = spiked
        s -= decay * s
        idx = np.nonzero(spiked)[0]
        if idx.size:
            s[idx, :] += conn[idx, :]
    spikes_count = spike_mat.sum(axis=0).astype(float)
    spike_times = [t[spike_mat[:, j]].tolist() for j in range(N)]
    return t, V, spikes_count, spike_times

# ----------------------------------------------------------------------------
# Honest spectral analysis
# ----------------------------------------------------------------------------
def spectral_metrics(sig, dt_ms):
    # Welch PSD (smooths the periodogram's high bin variance) + a fitted 1/f
    # aperiodic background. Prominence = how far the largest in-band peak rises
    # above that background, in dB. Peak frequency is reported so that an
    # artifactual "peak" pinned at a band edge is visible rather than hidden.
    sig = np.asarray(sig, float); sig = sig - np.mean(sig)
    fs = 1000.0 / dt_ms
    nperseg = int(min(len(sig), 50000))
    f, P = welch(sig, fs=fs, nperseg=nperseg)
    fit_mask = (f >= 1) & (f <= 100) & (P > 0)
    fb, Pb = f[fit_mask], P[fit_mask]
    b, a = np.polyfit(np.log10(fb), np.log10(Pb), 1)
    total = np.trapezoid(Pb, fb)
    def band(lo, hi):
        bmask = (fb >= lo) & (fb <= hi)
        if not np.any(bmask): return 0.0, 0.0, float("nan")
        bg = 10 ** (a + b*np.log10(fb[bmask]))
        ratio = Pb[bmask] / bg
        k = int(np.argmax(ratio))
        rel = float(np.trapezoid(Pb[bmask], fb[bmask]) / total) if total > 0 else 0.0
        return rel, float(10*np.log10(ratio[k])), float(fb[bmask][k])
    th = band(4, 8); ga = band(30, 80)
    return {"theta_rel": th[0], "theta_prom_db": th[1], "theta_fpk": th[2],
            "gamma_rel": ga[0], "gamma_prom_db": ga[1], "gamma_fpk": ga[2],
            "aperiodic_slope": float(b)}

# ============================================================================
# Configuration
# ============================================================================
t_span = [0, 1000]
dt = 0.01
num_runs = 5
I_mean_hh = 10.0
I_mean_lif = 1.0
BASE_SEED = 100

def summ(v): return [float(np.mean(v)), float(np.std(v))]

# --- block runners: each returns per-run metrics + a light 'last' payload ---
def block_hh(seed):
    s = time.time(); t, V = hh_single_euler(t_span, dt, I_mean_hh, noise_sigma=1.0, seed=seed); ct = time.time()-s
    peaks, _ = find_peaks(V, height=0)
    return {"rate": len(peaks)/(t_span[1]/1000.0), "time": ct, "spec": spectral_metrics(V, dt),
            "light": {"t": t, "V": V}}

def block_lif(seed, dtl=0.01):
    s = time.time(); t, V, sp = lif_single_euler(t_span, dtl, I_mean_lif, noise_sigma=0.1, seed=seed); ct = time.time()-s
    return {"rate": len(sp)/(t_span[1]/1000.0), "time": ct, "spec": spectral_metrics(V, dtl),
            "light": {"t": t, "V": V}}

def make_net(N_exc, N_inh, p_conn, hybrid=False):
    def run(seed):
        s = time.time()
        t, V, sc, st = _network(t_span, dt, N_exc, N_inh, p_conn, I_mean_lif, hybrid, seed=seed)
        ct = time.time()-s
        LFP = np.mean(V, axis=1)
        return {"rate": float(np.mean(sc)/(t_span[1]/1000.0)), "time": ct,
                "spec": spectral_metrics(LFP, dt),
                "light": {"t": t, "LFP": LFP, "spike_times": st}}
    return run

BLOCKS = [
    ("HH (Single)",                        block_hh),
    ("LIF (Single, dt=0.01)",              lambda sd: block_lif(sd, 0.01)),
    ("LIF (Single, Optimized dt=0.02)",    lambda sd: block_lif(sd, 0.02)),
    ("LIF Network (2%)",                   make_net(8, 2, 0.02)),
    ("LIF Network (Sparsified, 1.4%)",     make_net(8, 2, 0.014)),
    ("LIF Network (Scaled, 2%)",           make_net(80, 20, 0.02)),
    ("LIF Network (Scaled Sparsified, 1.4%)", make_net(80, 20, 0.014)),
    ("Hybrid Network (2%)",                make_net(8, 2, 0.02, hybrid=True)),
]

def ckpt_path(name):
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("%", "pct").replace("=", "").replace(".", "p")
    return os.path.join(CKPT, safe + ".pkl")

def run_all():
    for name, fn in BLOCKS:
        p = ckpt_path(name)
        if os.path.exists(p):
            print(f"skip (done): {name}", flush=True); continue
        t0 = time.time()
        rates=[]; times=[]; thr=[]; thp=[]; gar=[]; gap=[]; slope=[]; thf=[]; gaf=[]; last=None
        for r in range(num_runs):
            d = fn(BASE_SEED + r)
            rates.append(d["rate"]); times.append(d["time"])
            thr.append(d["spec"]["theta_rel"]); thp.append(d["spec"]["theta_prom_db"])
            gar.append(d["spec"]["gamma_rel"]); gap.append(d["spec"]["gamma_prom_db"])
            thf.append(d["spec"]["theta_fpk"]); gaf.append(d["spec"]["gamma_fpk"])
            slope.append(d["spec"]["aperiodic_slope"]); last = d
        res = {"name": name, "rate": summ(rates), "time": summ(times),
               "theta_rel": summ(thr), "theta_prom": summ(thp), "theta_fpk": summ(thf),
               "gamma_rel": summ(gar), "gamma_prom": summ(gap), "gamma_fpk": summ(gaf),
               "slope": summ(slope), "last": last}
        with open(p, "wb") as fh: pickle.dump(res, fh)
        print(f"done: {name}  ({time.time()-t0:.1f}s, rate={res['rate'][0]:.1f}Hz, "
              f"theta_prom={res['theta_prom'][0]:.1f}dB, gamma_prom={res['gamma_prom'][0]:.1f}dB)", flush=True)

def all_done():
    return all(os.path.exists(ckpt_path(n)) for n, _ in BLOCKS)

def load_all():
    out = {}
    for name, _ in BLOCKS:
        with open(ckpt_path(name), "rb") as fh:
            out[name] = pickle.load(fh)
    return out

def build_outputs():
    R = load_all()
    hdr = ("Model Type\tRate (Hz)\tTime (s)\tTheta pk(Hz)\tTheta prom(dB)"
           "\tGamma pk(Hz)\tGamma prom(dB)")
    lines = [hdr]
    for name, _ in BLOCKS:
        r = R[name]
        lines.append(f"{name}\t{r['rate'][0]:.1f} ± {r['rate'][1]:.1f}"
                     f"\t{r['time'][0]:.3f} ± {r['time'][1]:.3f}"
                     f"\t{r['theta_fpk'][0]:.1f}\t{r['theta_prom'][0]:.1f}"
                     f"\t{r['gamma_fpk'][0]:.1f}\t{r['gamma_prom'][0]:.1f}")
    table = "\n".join(lines)
    with open(os.path.join(OUTDIR, "results_v3.tsv"), "w") as fh: fh.write(table + "\n")
    print("\n" + table)

    def psd(sig, dt_ms):
        sig = np.asarray(sig, float) - np.mean(sig); n = len(sig); half = n//2
        return fftfreq(n, dt_ms/1000.0)[:half], np.abs(fft(sig)[:half])**2

    hh = R["HH (Single)"]["last"]["light"]
    lif = R["LIF (Single, dt=0.01)"]["last"]["light"]
    net = R["LIF Network (2%)"]["last"]["light"]
    scaled = R["LIF Network (Scaled, 2%)"]["last"]["light"]
    hybrid = R["Hybrid Network (2%)"]["last"]["light"]

    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1); plt.plot(hh["t"], hh["V"], lw=0.6, label='HH'); plt.ylabel('mV'); plt.legend()
    plt.title('Membrane Potential (Single Neuron, theta-modulated + noisy drive)')
    plt.subplot(2,1,2); plt.plot(lif["t"], lif["V"], lw=0.6, color='tab:orange', label='LIF')
    plt.xlabel('Time (ms)'); plt.ylabel('mV'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,'figure1_voltage_traces.png'), dpi=150); plt.close()

    plt.figure(figsize=(10, 6))
    fr,P = psd(hh["V"], dt); plt.subplot(2,1,1); plt.plot(fr,10*np.log10(P+1e-12),label='HH')
    plt.axvspan(4,8,color='y',alpha=0.3); plt.xlim(0,100); plt.ylabel('dB'); plt.legend()
    plt.title('Single-Neuron Power Spectra (theta band shaded)')
    fr,P = psd(lif["V"], dt); plt.subplot(2,1,2); plt.plot(fr,10*np.log10(P+1e-12),color='tab:orange',label='LIF')
    plt.axvspan(4,8,color='y',alpha=0.3); plt.xlim(0,100); plt.xlabel('Hz'); plt.ylabel('dB'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,'figure2_power_spectra.png'), dpi=150); plt.close()

    plt.figure(figsize=(10, 6)); plt.eventplot(net["spike_times"], colors='b')
    plt.title('Spike Raster (LIF Network, 8 exc + 2 inh, 2%)'); plt.xlabel('Time (ms)'); plt.ylabel('Neuron')
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,'figure3_raster.png'), dpi=150); plt.close()

    def specfig(light, title, fname):
        fr,P = psd(light["LFP"], dt)
        plt.figure(figsize=(10,6)); plt.plot(fr,10*np.log10(P+1e-12),label='LFP power')
        plt.axvspan(4,8,color='y',alpha=0.3,label='Theta'); plt.axvspan(30,80,color='g',alpha=0.3,label='Gamma')
        plt.title(title); plt.xlim(0,100); plt.xlabel('Hz'); plt.ylabel('dB'); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,fname), dpi=150); plt.close()
    specfig(net,'Network LFP Power Spectrum (10 neurons)','figure4_network_power.png')
    specfig(scaled,'Scaled Network LFP Power Spectrum (100 neurons)','figure5_scaled_network_power.png')
    specfig(hybrid,'Hybrid Network LFP Power Spectrum (10 neurons)','figure6_hybrid_network_power.png')
    print("\nAll figures written to", OUTDIR)

if __name__ == "__main__":
    run_all()
    if all_done():
        build_outputs()
        print("\nALL COMPLETE")
    else:
        remaining = [n for n,_ in BLOCKS if not os.path.exists(ckpt_path(n))]
        print("\nINCOMPLETE - remaining:", remaining)
