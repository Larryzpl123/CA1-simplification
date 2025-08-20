import numpy as np  # Library for numerical arrays and math functions
from scipy.integrate import solve_ivp  # For solving differential equations
import time  # Standard library for timing execution
from scipy.signal import find_peaks  # For detecting spikes in voltage traces
from scipy.fft import fft, fftfreq  # For computing Fast Fourier Transform
import matplotlib.pyplot as plt  # For generating figures

# Define voltage-dependent gating functions for Hodgkin-Huxley model
def alpha_m(V):
    den = 1 - np.exp(-(V + 40) / 10)
    return np.where(np.abs(den) < 1e-8, 1.0, 0.1 * (V + 40) / den)

def beta_m(V): return 4 * np.exp(-(V + 65) / 18)
def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))
def alpha_n(V):
    den = 1 - np.exp(-(V + 55) / 10)
    return np.where(np.abs(den) < 1e-8, 0.1, 0.01 * (V + 55) / den)
def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)

# ODE function for Hodgkin-Huxley neuron with time-dependent external current
def hh_ode(t, y, I_mean, g_Na=120, g_K=36, g_L=0.3, E_Na=50, E_K=-77, E_L=-54.4, C_m=1):
    V, m, h, n = y  # Unpack state variables: membrane potential and gating variables
    I_ext = I_mean + 2 * np.sin(2 * np.pi * 6 * t / 1000)  # Sinusoidal theta drive at 6 Hz
    dV = (-g_Na * m**3 * h * (V - E_Na) - g_K * n**4 * (V - E_K) - g_L * (V - E_L) + I_ext) / C_m  # Voltage derivative
    dm = alpha_m(V) * (1 - m) - beta_m(V) * m  # m gate derivative
    dh = alpha_h(V) * (1 - h) - beta_h(V) * h  # h gate derivative
    dn = alpha_n(V) * (1 - n) - beta_n(V) * n  # n gate derivative
    return [dV, dm, dh, dn]  # Return derivatives

# Simulation function for single Leaky Integrate-and-Fire neuron with time-dependent current
def lif_simulation(t_span, dt, tau=20, E_L=-65, R=10, V_th=-55, V_reset=-65, I_mean=0):
    t = np.arange(t_span[0], t_span[1], dt)  # Generate time array
    V = np.zeros(len(t))  # Initialize voltage array
    V[0] = E_L  # Set initial resting potential
    spikes = []  # List to store spike times
    for i in range(1, len(t)):  # Iterate over time steps
        I_ext = I_mean + 0.2 * np.sin(2 * np.pi * 6 * t[i] / 1000)  # Time-varying input (adjusted amplitude)
        V[i] = V[i-1] + dt * (-(V[i-1] - E_L) + R * I_ext) / tau  # Update voltage using Euler method
        if V[i] > V_th:  # Check for spike threshold
            V[i] = V_reset  # Reset voltage
            spikes.append(t[i])  # Record spike time
    return t, V, spikes  # Return time, voltage, and spikes

# Scaled network simulation for Leaky Integrate-and-Fire neurons (distinguishes exc and inh)
def lif_network_simulation(t_span, dt, N_exc=8, N_inh=2, p_conn=0.02, tau=20, E_L=-65, R=10, V_th=-55, V_reset=-65, I_mean=1.0, tau_syn=5, E_syn_exc=0, E_syn_inh=-80, seed=42):
    np.random.seed(seed)  # For reproducibility across runs
    base_g_exc = 0.05 * 8  # Original g_exc * original N_exc to keep mean input constant with size
    base_g_inh = 0.1 * 2   # Original g_inh * original N_inh
    N = N_exc + N_inh
    t = np.arange(t_span[0], t_span[1], dt)  # Generate time array
    V = np.zeros((len(t), N))  # Initialize voltage matrix
    V[0] = E_L * np.ones(N)  # Set initial resting potentials
    s = np.zeros((N, N))  # Current synaptic activations (pre, post); no time dimension for memory efficiency
    conn = (np.random.rand(N, N) < p_conn).astype(float)  # Connectivity matrix
    w_exc = base_g_exc * conn[:N_exc, :] / N_exc  # Scaled Excitatory weights (independent of p_conn)
    w_inh = base_g_inh * conn[N_exc:, :] / N_inh  # Scaled Inhibitory weights
    spikes_count = np.zeros(N)  # Spike counts per neuron
    spike_times = [[] for _ in range(N)]  # List of lists to store spike times per neuron
    for i in range(1, len(t)):  # Iterate over time steps
        I_ext = I_mean + 0.2 * np.sin(2 * np.pi * 6 * t[i] / 1000) * np.ones(N)  # External input
        gs_exc = np.sum(w_exc * s[:N_exc, :], axis=0)  # Sum over excitatory presynaptics
        gs_inh = np.sum(w_inh * s[N_exc:, :], axis=0)  # Sum over inhibitory presynaptics
        I_syn = gs_exc * (E_syn_exc - V[i-1]) + gs_inh * (E_syn_inh - V[i-1])  # Synaptic currents
        V[i] = V[i-1] + dt * (-(V[i-1] - E_L) + R * I_ext + R * I_syn) / tau  # Update voltages
        spiked = V[i] > V_th  # Detect spikes
        for neuron in range(N):  # Record spike times
            if spiked[neuron]:
                spike_times[neuron].append(t[i])
        V[i][spiked] = V_reset  # Reset spiked neurons
        spikes_count[spiked] += 1  # Count spikes
        s = s - dt * s / tau_syn  # Decay synaptic activations
        s += np.outer(spiked, np.ones(N)) * conn  # Add pulses for spiked neurons
    return t, V, spikes_count, spike_times  # Return time, voltages, spike counts, and spike times

# Hybrid network simulation (LIF for pyramidals, HH for interneurons)
def hybrid_network_simulation(t_span, dt, N_exc=8, N_inh=2, p_conn=0.02, tau=20, E_L=-65, R=10, V_th=-55, V_reset=-65, I_mean=1.0, tau_syn=5, E_syn_exc=0, E_syn_inh=-80, seed=42):
    np.random.seed(seed)  # For reproducibility
    base_g_exc = 0.05 * 8  # Original g_exc * original N_exc
    base_g_inh = 0.1 * 2   # Original g_inh * original N_inh
    N = N_exc + N_inh
    t = np.arange(t_span[0], t_span[1], dt)  # Generate time array
    V = np.zeros((len(t), N))  # Initialize voltage matrix
    V[0, :] = E_L  # Set initial resting potentials
    # For HH interneurons: gating variables
    m_inh = np.full(N_inh, 0.05)
    h_inh = np.full(N_inh, 0.6)
    n_inh = np.full(N_inh, 0.32)
    s = np.zeros((N, N))  # Current synaptic activations
    conn = (np.random.rand(N, N) < p_conn).astype(float)  # Connectivity matrix
    w_exc = base_g_exc * conn[:N_exc, :] / N_exc  # Scaled Excitatory weights
    w_inh = base_g_inh * conn[N_exc:, :] / N_inh  # Scaled Inhibitory weights
    spikes_count = np.zeros(N)  # Spike counts per neuron
    spike_times = [[] for _ in range(N)]  # List of lists to store spike times per neuron
    for i in range(1, len(t)):  # Iterate over time steps
        I_ext = I_mean + 0.2 * np.sin(2 * np.pi * 6 * t[i] / 1000) * np.ones(N)  # External input
        gs_exc = np.sum(w_exc * s[:N_exc, :], axis=0)  # Sum over excitatory presynaptics
        gs_inh = np.sum(w_inh * s[N_exc:, :], axis=0)  # Sum over inhibitory presynaptics
        I_syn = gs_exc * (E_syn_exc - V[i-1]) + gs_inh * (E_syn_inh - V[i-1])  # Synaptic currents
        # Update exc (LIF)
        V_exc_new = V[i-1, :N_exc] + dt * (-(V[i-1, :N_exc] - E_L) + R * I_ext[:N_exc] + R * I_syn[:N_exc]) / tau
        spiked_exc = V_exc_new > V_th
        V_exc_new[spiked_exc] = V_reset
        # Update inh (HH with Euler)
        V_inh = V[i-1, N_exc:]
        curr_inh = -120 * m_inh**3 * h_inh * (V_inh - 50) - 36 * n_inh**4 * (V_inh + 77) - 0.3 * (V_inh + 54.4) + I_ext[N_exc:] + I_syn[N_exc:]
        dV_inh = dt * curr_inh / 1.0
        dm_inh = dt * (alpha_m(V_inh) * (1 - m_inh) - beta_m(V_inh) * m_inh)
        dh_inh = dt * (alpha_h(V_inh) * (1 - h_inh) - beta_h(V_inh) * h_inh)
        dn_inh = dt * (alpha_n(V_inh) * (1 - n_inh) - beta_n(V_inh) * n_inh)
        V_inh_new = V_inh + dV_inh
        m_inh += dm_inh
        h_inh += dh_inh
        n_inh += dn_inh
        spiked_inh = V_inh_new > V_th
        V_inh_new[spiked_inh] = V_reset
        # Combine
        V[i, :N_exc] = V_exc_new
        V[i, N_exc:] = V_inh_new
        spiked = np.concatenate((spiked_exc, spiked_inh))
        spikes_count[spiked] += 1
        for neuron in range(N):
            if spiked[neuron]:
                spike_times[neuron].append(t[i])
        s = s - dt * s / tau_syn
        s += np.outer(spiked, np.ones(N)) * conn
    return t, V, spikes_count, spike_times

# Set simulation parameters
t_span = [0, 1000]  # Time span in ms
y0 = [-65, 0.05, 0.6, 0.32]  # Initial conditions for HH
I_mean_hh = 7  # Mean input for HH
I_mean_lif = 1.0  # Mean input for LIF
num_runs = 5  # Number of runs for statistics (as in paper)
dt_lif = 0.01  # dt for simulations (adjust to 0.02 if too slow)

# HH (Single)
spike_rates_hh = []
comp_times_hh = []
theta_dbs_hh = []
gamma_dbs_hh = []
for _ in range(num_runs):
    start = time.time()
    sol = solve_ivp(hh_ode, t_span, y0, args=(I_mean_hh,), method='LSODA', rtol=1e-6)
    comp_time = time.time() - start
    t, V = sol.t, sol.y[0]
    peaks, _ = find_peaks(V, height=0)
    rate = len(peaks) / (t_span[1] / 1000)
    N = len(t)
    yf = fft(V - np.mean(V))
    xf = fftfreq(N, t[1] - t[0])
    power = np.abs(yf[:N//2])**2
    mask_theta = (xf[:N//2] >= 4) & (xf[:N//2] <= 8)
    theta_power = np.max(power[mask_theta]) if np.any(mask_theta) else 0
    theta_db = 10 * np.log10(theta_power) if theta_power > 0 else 0
    mask_gamma = (xf[:N//2] >= 30) & (xf[:N//2] <= 80)
    gamma_power = np.max(power[mask_gamma]) if np.any(mask_gamma) else 0
    gamma_db = 10 * np.log10(gamma_power) if gamma_power > 0 else 0
    spike_rates_hh.append(rate)
    comp_times_hh.append(comp_time)
    theta_dbs_hh.append(theta_db)
    gamma_dbs_hh.append(gamma_db)
mean_rate_hh = np.mean(spike_rates_hh)
sd_rate_hh = np.std(spike_rates_hh)
mean_time_hh = np.mean(comp_times_hh)
sd_time_hh = np.std(comp_times_hh)
mean_theta_hh = np.mean(theta_dbs_hh)
mean_gamma_hh = np.mean(gamma_dbs_hh)

# LIF (Single, dt=0.01)
spike_rates_lif = []
comp_times_lif = []
theta_dbs_lif = []
gamma_dbs_lif = []
for _ in range(num_runs):
    start = time.time()
    t, V, spikes = lif_simulation(t_span, dt_lif, I_mean=I_mean_lif)
    comp_time = time.time() - start
    rate = len(spikes) / (t_span[1] / 1000)
    N = len(t)
    yf = fft(V - np.mean(V))
    xf = fftfreq(N, dt_lif)
    power = np.abs(yf[:N//2])**2
    mask_theta = (xf[:N//2] >= 4) & (xf[:N//2] <= 8)
    theta_power = np.max(power[mask_theta]) if np.any(mask_theta) else 0
    theta_db = 10 * np.log10(theta_power) if theta_power > 0 else 0
    mask_gamma = (xf[:N//2] >= 30) & (xf[:N//2] <= 80)
    gamma_power = np.max(power[mask_gamma]) if np.any(mask_gamma) else 0
    gamma_db = 10 * np.log10(gamma_power) if gamma_power > 0 else 0
    spike_rates_lif.append(rate)
    comp_times_lif.append(comp_time)
    theta_dbs_lif.append(theta_db)
    gamma_dbs_lif.append(gamma_db)
mean_rate_lif = np.mean(spike_rates_lif)
sd_rate_lif = np.std(spike_rates_lif)
mean_time_lif = np.mean(comp_times_lif)
sd_time_lif = np.std(comp_times_lif)
mean_theta_lif = np.mean(theta_dbs_lif)
mean_gamma_lif = np.mean(gamma_dbs_lif)

# LIF (Single, Optimized dt=0.02)
dt_lif_opt = 0.02
spike_rates_lif_opt = []
comp_times_lif_opt = []
theta_dbs_lif_opt = []
gamma_dbs_lif_opt = []
for _ in range(num_runs):
    start = time.time()
    t, V, spikes = lif_simulation(t_span, dt_lif_opt, I_mean=I_mean_lif)
    comp_time = time.time() - start
    rate = len(spikes) / (t_span[1] / 1000)
    N = len(t)
    yf = fft(V - np.mean(V))
    xf = fftfreq(N, dt_lif_opt)
    power = np.abs(yf[:N//2])**2
    mask_theta = (xf[:N//2] >= 4) & (xf[:N//2] <= 8)
    theta_power = np.max(power[mask_theta]) if np.any(mask_theta) else 0
    theta_db = 10 * np.log10(theta_power) if theta_power > 0 else 0
    mask_gamma = (xf[:N//2] >= 30) & (xf[:N//2] <= 80)
    gamma_power = np.max(power[mask_gamma]) if np.any(mask_gamma) else 0
    gamma_db = 10 * np.log10(gamma_power) if gamma_power > 0 else 0
    spike_rates_lif_opt.append(rate)
    comp_times_lif_opt.append(comp_time)
    theta_dbs_lif_opt.append(theta_db)
    gamma_dbs_lif_opt.append(gamma_db)
mean_rate_lif_opt = np.mean(spike_rates_lif_opt)
sd_rate_lif_opt = np.std(spike_rates_lif_opt)
mean_time_lif_opt = np.mean(comp_times_lif_opt)
sd_time_lif_opt = np.std(comp_times_lif_opt)
mean_theta_lif_opt = np.mean(theta_dbs_lif_opt)
mean_gamma_lif_opt = np.mean(gamma_dbs_lif_opt)

# LIF Network (2%)
spike_rates_net2 = []
comp_times_net2 = []
theta_dbs_net2 = []
gamma_dbs_net2 = []
for run in range(num_runs):
    start = time.time()
    t, V, spikes_count, spike_times = lif_network_simulation(t_span, dt_lif, N_exc=8, N_inh=2, p_conn=0.02, I_mean=I_mean_lif, seed=42 + run)  # Vary seed per run for variability
    comp_time = time.time() - start
    mean_rate = np.mean(spikes_count) / (t_span[1] / 1000)
    LFP = np.mean(V, axis=1)
    N = len(t)
    yf = fft(LFP - np.mean(LFP))
    xf = fftfreq(N, dt_lif)
    power = np.abs(yf[:N//2])**2
    mask_theta = (xf[:N//2] >= 4) & (xf[:N//2] <= 8)
    theta_power = np.max(power[mask_theta]) if np.any(mask_theta) else 0
    theta_db = 10 * np.log10(theta_power) if theta_power > 0 else 0
    mask_gamma = (xf[:N//2] >= 30) & (xf[:N//2] <= 80)
    gamma_power = np.max(power[mask_gamma]) if np.any(mask_gamma) else 0
    gamma_db = 10 * np.log10(gamma_power) if gamma_power > 0 else 0
    spike_rates_net2.append(mean_rate)
    comp_times_net2.append(comp_time)
    theta_dbs_net2.append(theta_db)
    gamma_dbs_net2.append(gamma_db)
mean_rate_net2 = np.mean(spike_rates_net2)
sd_rate_net2 = np.std(spike_rates_net2)
mean_time_net2 = np.mean(comp_times_net2)
sd_time_net2 = np.std(comp_times_net2)
mean_theta_net2 = np.mean(theta_dbs_net2)
mean_gamma_net2 = np.mean(gamma_dbs_net2)

# LIF Network (Sparsified, 1.4%)
spike_rates_net14 = []
comp_times_net14 = []
theta_dbs_net14 = []
gamma_dbs_net14 = []
for run in range(num_runs):
    start = time.time()
    t, V, spikes_count, spike_times = lif_network_simulation(t_span, dt_lif, N_exc=8, N_inh=2, p_conn=0.014, I_mean=I_mean_lif, seed=42 + run)
    comp_time = time.time() - start
    mean_rate = np.mean(spikes_count) / (t_span[1] / 1000)
    LFP = np.mean(V, axis=1)
    N = len(t)
    yf = fft(LFP - np.mean(LFP))
    xf = fftfreq(N, dt_lif)
    power = np.abs(yf[:N//2])**2
    mask_theta = (xf[:N//2] >= 4) & (xf[:N//2] <= 8)
    theta_power = np.max(power[mask_theta]) if np.any(mask_theta) else 0
    theta_db = 10 * np.log10(theta_power) if theta_power > 0 else 0
    mask_gamma = (xf[:N//2] >= 30) & (xf[:N//2] <= 80)
    gamma_power = np.max(power[mask_gamma]) if np.any(mask_gamma) else 0
    gamma_db = 10 * np.log10(gamma_power) if gamma_power > 0 else 0
    spike_rates_net14.append(mean_rate)
    comp_times_net14.append(comp_time)
    theta_dbs_net14.append(theta_db)
    gamma_dbs_net14.append(gamma_db)
mean_rate_net14 = np.mean(spike_rates_net14)
sd_rate_net14 = np.std(spike_rates_net14)
mean_time_net14 = np.mean(comp_times_net14)
sd_time_net14 = np.std(comp_times_net14)
mean_theta_net14 = np.mean(theta_dbs_net14)
mean_gamma_net14 = np.mean(gamma_dbs_net14)

# LIF Network (Scaled, 2%)
spike_rates_scaled2 = []
comp_times_scaled2 = []
theta_dbs_scaled2 = []
gamma_dbs_scaled2 = []
for run in range(num_runs):
    start = time.time()
    t, V, spikes_count, spike_times = lif_network_simulation(t_span, dt_lif, N_exc=80, N_inh=20, p_conn=0.02, I_mean=I_mean_lif, seed=42 + run)
    comp_time = time.time() - start
    mean_rate = np.mean(spikes_count) / (t_span[1] / 1000)
    LFP = np.mean(V, axis=1)
    N = len(t)
    yf = fft(LFP - np.mean(LFP))
    xf = fftfreq(N, dt_lif)
    power = np.abs(yf[:N//2])**2
    mask_theta = (xf[:N//2] >= 4) & (xf[:N//2] <= 8)
    theta_power = np.max(power[mask_theta]) if np.any(mask_theta) else 0
    theta_db = 10 * np.log10(theta_power) if theta_power > 0 else 0
    mask_gamma = (xf[:N//2] >= 30) & (xf[:N//2] <= 80)
    gamma_power = np.max(power[mask_gamma]) if np.any(mask_gamma) else 0
    gamma_db = 10 * np.log10(gamma_power) if gamma_power > 0 else 0
    spike_rates_scaled2.append(mean_rate)
    comp_times_scaled2.append(comp_time)
    theta_dbs_scaled2.append(theta_db)
    gamma_dbs_scaled2.append(gamma_db)
mean_rate_scaled2 = np.mean(spike_rates_scaled2)
sd_rate_scaled2 = np.std(spike_rates_scaled2)
mean_time_scaled2 = np.mean(comp_times_scaled2)
sd_time_scaled2 = np.std(comp_times_scaled2)
mean_theta_scaled2 = np.mean(theta_dbs_scaled2)
mean_gamma_scaled2 = np.mean(gamma_dbs_scaled2)

# LIF Network (Scaled Sparsified, 1.4%)
spike_rates_scaled14 = []
comp_times_scaled14 = []
theta_dbs_scaled14 = []
gamma_dbs_scaled14 = []
for run in range(num_runs):
    start = time.time()
    t, V, spikes_count, spike_times = lif_network_simulation(t_span, dt_lif, N_exc=80, N_inh=20, p_conn=0.014, I_mean=I_mean_lif, seed=42 + run)
    comp_time = time.time() - start
    mean_rate = np.mean(spikes_count) / (t_span[1] / 1000)
    LFP = np.mean(V, axis=1)
    N = len(t)
    yf = fft(LFP - np.mean(LFP))
    xf = fftfreq(N, dt_lif)
    power = np.abs(yf[:N//2])**2
    mask_theta = (xf[:N//2] >= 4) & (xf[:N//2] <= 8)
    theta_power = np.max(power[mask_theta]) if np.any(mask_theta) else 0
    theta_db = 10 * np.log10(theta_power) if theta_power > 0 else 0
    mask_gamma = (xf[:N//2] >= 30) & (xf[:N//2] <= 80)
    gamma_power = np.max(power[mask_gamma]) if np.any(mask_gamma) else 0
    gamma_db = 10 * np.log10(gamma_power) if gamma_power > 0 else 0
    spike_rates_scaled14.append(mean_rate)
    comp_times_scaled14.append(comp_time)
    theta_dbs_scaled14.append(theta_db)
    gamma_dbs_scaled14.append(gamma_db)
mean_rate_scaled14 = np.mean(spike_rates_scaled14)
sd_rate_scaled14 = np.std(spike_rates_scaled14)
mean_time_scaled14 = np.mean(comp_times_scaled14)
sd_time_scaled14 = np.std(comp_times_scaled14)
mean_theta_scaled14 = np.mean(theta_dbs_scaled14)
mean_gamma_scaled14 = np.mean(gamma_dbs_scaled14)

# Hybrid Network (2%)
spike_rates_hybrid2 = []
comp_times_hybrid2 = []
theta_dbs_hybrid2 = []
gamma_dbs_hybrid2 = []
for run in range(num_runs):
    start = time.time()
    t, V, spikes_count, spike_times = hybrid_network_simulation(t_span, dt_lif, N_exc=8, N_inh=2, p_conn=0.02, I_mean=I_mean_lif, seed=42 + run)
    comp_time = time.time() - start
    mean_rate = np.mean(spikes_count) / (t_span[1] / 1000)
    LFP = np.mean(V, axis=1)
    N = len(t)
    yf = fft(LFP - np.mean(LFP))
    xf = fftfreq(N, dt_lif)
    power = np.abs(yf[:N//2])**2
    mask_theta = (xf[:N//2] >= 4) & (xf[:N//2] <= 8)
    theta_power = np.max(power[mask_theta]) if np.any(mask_theta) else 0
    theta_db = 10 * np.log10(theta_power) if theta_power > 0 else 0
    mask_gamma = (xf[:N//2] >= 30) & (xf[:N//2] <= 80)
    gamma_power = np.max(power[mask_gamma]) if np.any(mask_gamma) else 0
    gamma_db = 10 * np.log10(gamma_power) if gamma_power > 0 else 0
    spike_rates_hybrid2.append(mean_rate)
    comp_times_hybrid2.append(comp_time)
    theta_dbs_hybrid2.append(theta_db)
    gamma_dbs_hybrid2.append(gamma_db)
mean_rate_hybrid2 = np.mean(spike_rates_hybrid2)
sd_rate_hybrid2 = np.std(spike_rates_hybrid2)
mean_time_hybrid2 = np.mean(comp_times_hybrid2)
sd_time_hybrid2 = np.std(comp_times_hybrid2)
mean_theta_hybrid2 = np.mean(theta_dbs_hybrid2)
mean_gamma_hybrid2 = np.mean(gamma_dbs_hybrid2)

# Generate figures using single runs (comment out if not needed)
sol_hh = solve_ivp(hh_ode, t_span, y0, args=(I_mean_hh,), method='LSODA', rtol=1e-6)
t_hh, V_hh = sol_hh.t, sol_hh.y[0]
N_hh = len(t_hh)
yf_hh = fft(V_hh - np.mean(V_hh))
xf_hh = fftfreq(N_hh, t_hh[1] - t_hh[0])
power_hh = np.abs(yf_hh[:N_hh//2])**2

t_lif, V_lif, spikes_lif = lif_simulation(t_span, dt_lif, I_mean=I_mean_lif)
N_lif = len(t_lif)
yf_lif = fft(V_lif - np.mean(V_lif))
xf_lif = fftfreq(N_lif, dt_lif)
power_lif = np.abs(yf_lif[:N_lif//2])**2

t_net, V_net, spikes_net, spike_times_net = lif_network_simulation(t_span, dt_lif, N_exc=8, N_inh=2, p_conn=0.02, I_mean=I_mean_lif, seed=42)
LFP_net = np.mean(V_net, axis=1)
N_net = len(t_net)
yf_net = fft(LFP_net - np.mean(LFP_net))
xf_net = fftfreq(N_net, dt_lif)
power_net = np.abs(yf_net[:N_net//2])**2

t_scaled, V_scaled, spikes_scaled, spike_times_scaled = lif_network_simulation(t_span, dt_lif, N_exc=80, N_inh=20, p_conn=0.02, I_mean=I_mean_lif, seed=42)
LFP_scaled = np.mean(V_scaled, axis=1)
yf_scaled = fft(LFP_scaled - np.mean(LFP_scaled))
xf_scaled = fftfreq(len(t_scaled), dt_lif)
power_scaled = np.abs(yf_scaled[:len(t_scaled)//2])**2

t_hybrid, V_hybrid, spikes_hybrid, spike_times_hybrid = hybrid_network_simulation(t_span, dt_lif, N_exc=8, N_inh=2, p_conn=0.02, I_mean=I_mean_lif, seed=42)
LFP_hybrid = np.mean(V_hybrid, axis=1)
yf_hybrid = fft(LFP_hybrid - np.mean(LFP_hybrid))
xf_hybrid = fftfreq(len(t_hybrid), dt_lif)
power_hybrid = np.abs(yf_hybrid[:len(t_hybrid)//2])**2

# Figure 1: Voltage traces for HH and LIF single
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t_hh, V_hh, label='HH Model')
plt.title('Membrane Potential Dynamics (Single Neuron)')
plt.ylabel('Potential (mV)')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t_lif, V_lif, label='LIF Model')
plt.xlabel('Time (ms)')
plt.ylabel('Potential (mV)')
plt.legend()
plt.tight_layout()
plt.savefig('figure1_voltage_traces.png')

# Figure 2: Power spectra for HH and LIF single
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(xf_hh[:N_hh//2], 10 * np.log10(power_hh), label='HH Power Spectrum')
plt.title('Power Spectra (Theta Band Highlighted)')
plt.xlim(0, 10)
plt.ylabel('Power (dB)')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(xf_lif[:N_lif//2], 10 * np.log10(power_lif), label='LIF Power Spectrum')
plt.xlabel('Frequency (Hz)')
plt.xlim(0, 10)
plt.ylabel('Power (dB)')
plt.legend()
plt.tight_layout()
plt.savefig('figure2_power_spectra.png')

# Figure 3: Spike Raster Plot for LIF network (2% connectivity)
plt.figure(figsize=(10, 6))
plt.eventplot(spike_times_net, orientation='horizontal', colors='b')
plt.title('Spike Raster Plot (Network, 2% Connectivity)')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.tight_layout()
plt.savefig('figure3_raster.png')

# Figure 4: Network Power Spectrum for LIF network
plt.figure(figsize=(10, 6))
plt.plot(xf_net[:N_net//2], 10 * np.log10(power_net), label='Network Power Spectrum')
plt.axvspan(4, 8, color='yellow', alpha=0.3, label='Theta Band')
plt.axvspan(30, 80, color='green', alpha=0.3, label='Gamma Band')
plt.title('Power Spectrum of Network Local Field Potential')
plt.xlim(0, 100)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.legend()
plt.tight_layout()
plt.savefig('figure4_network_power.png')

# Figure 5: Power Spectrum for Scaled LIF network
plt.figure(figsize=(10, 6))
plt.plot(xf_scaled[:len(t_scaled)//2], 10 * np.log10(power_scaled), label='Scaled Network Power Spectrum')
plt.axvspan(4, 8, color='yellow', alpha=0.3, label='Theta Band')
plt.axvspan(30, 80, color='green', alpha=0.3, label='Gamma Band')
plt.title('Power Spectrum of Scaled Network Local Field Potential')
plt.xlim(0, 100)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.legend()
plt.tight_layout()
plt.savefig('figure5_scaled_network_power.png')

# Figure 6: Power Spectrum for Hybrid network
plt.figure(figsize=(10, 6))
plt.plot(xf_hybrid[:len(t_hybrid)//2], 10 * np.log10(power_hybrid), label='Hybrid Network Power Spectrum')
plt.axvspan(4, 8, color='yellow', alpha=0.3, label='Theta Band')
plt.axvspan(30, 80, color='green', alpha=0.3, label='Gamma Band')
plt.title('Power Spectrum of Hybrid Network Local Field Potential')
plt.xlim(0, 100)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.legend()
plt.tight_layout()
plt.savefig('figure6_hybrid_network_power.png')

# Output results table
print("Model Type\tSpiking Rate (Hz, mean ± SD)\tComputation Time (s, mean ± SD)\tTheta Power (dB)\tGamma Power (dB)")
print(f"HH (Single)\t{mean_rate_hh:.1f} ± {sd_rate_hh:.1f}\t{mean_time_hh:.3f} ± {sd_time_hh:.3f}\t{mean_theta_hh:.1f}\t{mean_gamma_hh:.1f}")
print(f"LIF (Single, dt=0.01)\t{mean_rate_lif:.1f} ± {sd_rate_lif:.1f}\t{mean_time_lif:.3f} ± {sd_time_lif:.3f}\t{mean_theta_lif:.1f}\t{mean_gamma_lif:.1f}")
print(f"LIF (Single, Optimized dt=0.02)\t{mean_rate_lif_opt:.1f} ± {sd_rate_lif_opt:.1f}\t{mean_time_lif_opt:.3f} ± {sd_time_lif_opt:.3f}\t{mean_theta_lif_opt:.1f}\t{mean_gamma_lif_opt:.1f}")
print(f"LIF Network (2%)\t{mean_rate_net2:.1f} ± {sd_rate_net2:.1f}\t{mean_time_net2:.3f} ± {sd_time_net2:.3f}\t{mean_theta_net2:.1f}\t{mean_gamma_net2:.1f}")
print(f"LIF Network (Sparsified, 1.4%)\t{mean_rate_net14:.1f} ± {sd_rate_net14:.1f}\t{mean_time_net14:.3f} ± {sd_time_net14:.3f}\t{mean_theta_net14:.1f}\t{mean_gamma_net14:.1f}")
print(f"LIF Network (Scaled, 2%)\t{mean_rate_scaled2:.1f} ± {sd_rate_scaled2:.1f}\t{mean_time_scaled2:.3f} ± {sd_time_scaled2:.3f}\t{mean_theta_scaled2:.1f}\t{mean_gamma_scaled2:.1f}")
print(f"LIF Network (Scaled Sparsified, 1.4%)\t{mean_rate_scaled14:.1f} ± {sd_rate_scaled14:.1f}\t{mean_time_scaled14:.3f} ± {sd_time_scaled14:.3f}\t{mean_theta_scaled14:.1f}\t{mean_gamma_scaled14:.1f}")
print(f"Hybrid Network (2%)\t{mean_rate_hybrid2:.1f} ± {sd_rate_hybrid2:.1f}\t{mean_time_hybrid2:.3f} ± {sd_time_hybrid2:.3f}\t{mean_theta_hybrid2:.1f}\t{mean_gamma_hybrid2:.1f}")
