#!/usr/bin/env python3
"""
spectral_null.py — a NULL DISTRIBUTION for peak prominence.
Larry (Peilin) Zhong

THE PROBLEM THIS FIXES
----------------------
The v3 metric reports "peak prominence" = how many dB the LARGEST bin inside a
band rises above a fitted 1/f background. That statistic has a POSITIVE BIAS.

Welch bins are chi-square distributed. Taking the max over ~40 noisy bins in the
30-80 Hz band lands several dB above the mean EVEN WHEN THERE IS NO RHYTHM. So
"5 dB of gamma prominence" is not evidence of gamma — it is roughly what pure
noise gives you. The null of this statistic is NOT 0 dB.

v3 had two artifact flags (band-edge, firing-rate harmonic). It had no flag for
this. Neither did the first version of the PING pilot: it reported ~5 dB of
"gamma" at every point of a grid in which the I->E in-degree varied 24-fold,
which is the signature of a statistic measuring its own bias.

WHAT THIS MODULE DOES
---------------------
Computes the same prominence statistic, but calibrates it against surrogates
that preserve everything EXCEPT the rhythm being tested:

  * jitter_surrogate   — displace each spike by U(-w, +w). Destroys temporal
                         structure above ~1/(2w); preserves rate and slow drift.
                         w is chosen from the band under test.

  * poisson_surrogate  — smooth the population rate (Gaussian, sigma chosen to
                         erase the band under test but keep slower rhythms),
                         then redraw spikes as an inhomogeneous Poisson process.
                         Preserves firing rate AND the 6 Hz theta modulation,
                         destroys fast synchrony. This is the right null for
                         "is the gamma EMERGENT?" — the surrogate network has
                         identical rates and identical theta, and no PING.

Report:  observed prominence, null mean, null 95th pct, z, and a permutation p.
A peak counts only if p < 0.05 against this null. Raw dB alone means nothing.

SELF-TEST (run this file directly)
----------------------------------
  case A: constant-rate Poisson spikes -> NO rhythm exists.
          The old metric still reports several dB. The null test must NOT call
          it significant. (If it does, the test is broken.)
  case B: spikes whose rate is modulated at 40 Hz -> a rhythm DOES exist.
          The null test must find it. (If it doesn't, the test has no power.)

Usage in the pilot:
    from spectral_null import test_rhythm
    res = test_rhythm(spike_times_s, n_neurons, T_s, band=(30,80))
    if res["p"] < 0.05: ...   # only now may you say "gamma"
"""

import numpy as np
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d

# ---- POPULATION-RATE BINNING AND FREQUENCY RESOLUTION -----------------------
# 1 ms binning => Nyquist 500 Hz, far above the 80 Hz we analyse. The old 0.1 ms
# binning bought nothing and cost 10x the compute; that compute is better spent
# on RECORDING LENGTH, which is what buys frequency resolution.
#
# WHY RESOLUTION IS NOW A FIRST-CLASS PARAMETER. The drive-harmonic screen
# rejects a band of +/-1.5*df around every multiple of the drive. With a drive
# at f0, harmonics are spaced f0 apart, so the fraction of the analysis band the
# screen destroys is
#         blocked  ~  3 * df / f0
# At df = 1.22 Hz and f0 = 6 Hz that is 61% of the band (measured: 66%), and a
# GENUINE 40 Hz rhythm is rejected as "7 x drive" because 42 - 40.3 = 1.7 Hz is
# inside the tolerance. The screen then destroys the thing it exists to protect.
# Requiring blocked <= X gives
#         df <= X * f0 / 3          (X = 0.10, f0 = 6 Hz  ->  df <= 0.20 Hz)
# which at 1 ms binning needs nperseg >= 5000 samples, i.e. >= 5 s per Welch
# segment and ~20 s of recording. Verified: at 20 s (df = 0.15 Hz, 8% blocked)
# the screen rejects a true harmonic AND keeps a genuine 40 Hz gamma.
DT_S = 1e-3          # 1 ms binning for the population rate
NPERSEG = 8192       # with 1 ms bins -> df = 1000/8192 = 0.12 Hz
FIT_LO, FIT_HI = 1.0, 100.0


def required_df(target_blocked_fraction, drive_hz):
    """Max frequency resolution (Hz) that keeps the drive-harmonic screen from
    destroying more than `target_blocked_fraction` of the analysis band."""
    return target_blocked_fraction * drive_hz / 3.0


def blocked_fraction(df, drive_hz, band=(30.0, 80.0)):
    """Fraction of `band` that the drive-harmonic screen rejects at resolution df."""
    tol = 1.5 * df
    lo, hi = band
    harm = [drive_hz * k for k in range(1, 200) if lo - tol <= drive_hz * k <= hi + tol]
    blocked = sum(max(0.0, min(hi, h + tol) - max(lo, h - tol)) for h in harm)
    return min(blocked / (hi - lo), 1.0)


def check_resolution(df, drive_hz, band=(30.0, 80.0), warn_above=0.15):
    """Refuse to let a reader trust a harmonic screen that is eating the band."""
    frac = blocked_fraction(df, drive_hz, band)
    if frac > warn_above:
        print(f"  !! RESOLUTION WARNING: df = {df:.2f} Hz with a {drive_hz:g} Hz drive")
        print(f"     means the drive-harmonic screen rejects {100*frac:.0f}% of the "
              f"{band[0]:g}-{band[1]:g} Hz band.")
        print(f"     A GENUINE rhythm inside those zones would be rejected too.")
        print(f"     Need df <= {required_df(warn_above, drive_hz):.2f} Hz "
              f"-> lengthen the recording. See the note at the top of this file.")
    return frac


# ============================================================================
# VERSION BANNER — so two machines can never silently run different code again.
# Larry's run and his brother's run of ca1_v2 disagreed (gamma 8.51 dB vs 3.17 dB,
# null 6.68 vs 1.61, verdict "GAMMA" vs "no gamma") because they were executing
# different files. Neither table meant anything. Print the fingerprint, always.
# ============================================================================
def _fingerprint():
    import hashlib, os
    out = {}
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in ("spectral_null.py", "ca1_v2.py", "artifact_demo.py"):
        p = os.path.join(here, fn)
        out[fn] = (hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
                   if os.path.exists(p) else "MISSING")
    return out


def print_version_banner(extra=None):
    fp = _fingerprint()
    print("=" * 78)
    print("VERSION FINGERPRINT  —  these must match on every machine")
    print("=" * 78)
    for k, v in fp.items():
        print(f"  {k:<22} sha256[:12] = {v}")
    if extra:
        for k, v in extra.items():
            print(f"  {k:<22} = {v}")
    print("=" * 78)
    return fp


# ---------------------------------------------------------------- statistic
def pop_rate(spike_times_s, n_neurons, T_s, dt_s=DT_S):
    """Population firing rate (Hz) from a flat list of spike times."""
    edges = np.arange(0.0, T_s + dt_s, dt_s)
    counts, _ = np.histogram(np.asarray(spike_times_s), edges)
    return counts / (n_neurons * dt_s)


def psd(sig, fs, nperseg=NPERSEG):
    sig = np.asarray(sig, float)
    sig = sig - sig.mean()
    f, P = welch(sig, fs=fs, nperseg=int(min(len(sig), nperseg)))
    return f, P


def prominence(f, P, band, fit_lo=FIT_LO, fit_hi=FIT_HI):
    """dB of the largest in-band bin above a log-log-fitted aperiodic background.
    This is exactly the v3 statistic — reproduced faithfully so the null we build
    is the null OF THAT STATISTIC, not of some other one."""
    m = (f >= fit_lo) & (f <= fit_hi) & (P > 0)
    fb, Pb = f[m], P[m]
    if len(fb) < 10:
        return 0.0, np.nan, np.nan
    b, a = np.polyfit(np.log10(fb), np.log10(Pb), 1)
    bg = 10 ** (a + b * np.log10(fb))
    bm = (fb >= band[0]) & (fb <= band[1])
    if not np.any(bm):
        return 0.0, np.nan, float(b)
    ratio = Pb[bm] / bg[bm]
    k = int(np.argmax(ratio))
    return float(10 * np.log10(ratio[k])), float(fb[bm][k]), float(b)


def prominence_notched(f, P, band, drive_hz, notch_bins=4.0,
                       fit_lo=FIT_LO, fit_hi=FIT_HI):
    """THE FIX. Excise the drive's harmonic comb from the spectrum BEFORE taking
    the max, then fit the background and take the max over what is left.

    WHY THE UN-NOTCHED STATISTIC MUST NOT BE USED ON A DRIVEN NETWORK.
    Measured on ground truth (union of two independent sources: a 6 Hz volley
    train at 5 ms jitter, plus optionally a genuine gamma volley train), 21 s,
    df = 0.12 Hz:

        case                        RAW max-in-band     COMB-NOTCHED
        tight theta, NO gamma       33.23 dB @ 30.0     6.46 dB @ 32.1  (null)
        tight theta + REAL 40 Hz    28.23 dB @ 30.0    28.81 dB @ 40.0
        tight theta + REAL 55 Hz    28.41 dB @ 30.0    23.65 dB @ 55.1
        tight theta + REAL 68 Hz    28.06 dB @ 30.0    22.25 dB @ 68.0

    In the RAW column every peak lands on the 30 Hz comb tooth and every value is
    28-33 dB. Whether a rhythm exists, and at what frequency, changes the answer
    not at all: the statistic is blind. The harmonic comb does not merely create
    false positives -- it MASKS real rhythms, creating false negatives too.

    Notching removes the comb: no-gamma data falls back to the null, and a real
    rhythm is recovered at its true frequency to within 0.1 Hz.

    NOTCH WIDTH = 4 BINS, and that number is measured, not chosen. A comb tooth
    LEAKS. At sigma = 3 ms locking, the power around the 30 Hz tooth is:
            +0 bins  +35.0 dB      +2 bins   +3.2 dB
            +1 bin   +25.6 dB      +3 bins   -0.9 dB
    so a 2-bin notch still leaves the shoulder inside the search region. Measured
    false-positive rate on gamma-free, sigma=3 ms data (the worst case):
            notch_bins = 2  ->  67% false positives
            notch_bins = 4  ->   0%
            notch_bins = 6  ->   0%
    4 bins is the smallest width that works; it excises ~18% of the 30-80 Hz band
    (vs 6% at 2 bins), and it does NOT remove genuine rhythms: real 40, 55 and 68
    Hz rhythms are all still recovered to within 0.1 Hz. Full validation, 6/6:
            no rhythm (loose 25 ms)      -> rejected   (0% FP)
            tight comb 8 ms, no gamma    -> rejected   (0% FP)
            tight comb 3 ms, no gamma    -> rejected   (0% FP)
            tight comb + REAL 40 Hz      -> RHYTHM @ 40.0 Hz  (0% FN)
            tight comb + REAL 55 Hz      -> RHYTHM @ 55.1 Hz  (0% FN)
            tight comb + REAL 68 Hz      -> RHYTHM @ 68.0 Hz  (0% FN)
    """
    keep = np.ones_like(f, dtype=bool)
    df = f[1] - f[0]
    for k in range(1, int(fit_hi / max(drive_hz, 1e-9)) + 2):
        keep &= np.abs(f - drive_hz * k) > notch_bins * df
    m = keep & (f >= fit_lo) & (f <= fit_hi) & (P > 0)
    fb, Pb = f[m], P[m]
    if len(fb) < 10:
        return 0.0, np.nan, np.nan
    b, a = np.polyfit(np.log10(fb), np.log10(Pb), 1)
    bg = 10 ** (a + b * np.log10(fb))
    bm = (fb >= band[0]) & (fb <= band[1])
    if not np.any(bm):
        return 0.0, np.nan, float(b)
    ratio = Pb[bm] / bg[bm]
    k = int(np.argmax(ratio))
    return float(10 * np.log10(ratio[k])), float(fb[bm][k]), float(b)


# ---------------------------------------------------------------- surrogates
def jitter_surrogate(spike_times_s, band, T_s, rng):
    """Displace each spike uniformly in +/- w, w = 1/(2*f_low).
    Destroys synchrony inside the band; leaves rate and slower rhythms intact."""
    w = 1.0 / (2.0 * band[0])                      # gamma(30-80) -> w = 16.7 ms
    st = np.asarray(spike_times_s) + rng.uniform(-w, w, size=len(spike_times_s))
    return np.clip(st, 0.0, T_s)


def poisson_surrogate(rate_hz, n_neurons, band, rng, dt_s=DT_S):
    """Inhomogeneous-Poisson null: keep the SLOW rate profile (so theta and the
    mean firing rate are matched exactly), erase fast synchrony, redraw spikes.
    sigma is set so the band under test is attenuated but slower bands survive."""
    sigma_s = 1.0 / (3.0 * band[0])                # gamma(30-80) -> sigma ~11 ms
    smooth = gaussian_filter1d(np.asarray(rate_hz, float), sigma_s / dt_s)
    smooth = np.clip(smooth, 0, None)
    lam = smooth * n_neurons * dt_s                # expected spikes / bin
    counts = rng.poisson(lam)
    return counts / (n_neurons * dt_s)


# ---------------------------------------------------------------- the test
def harmonic_of(fpk, f0, df, max_k=15, tol_bins=1.5):
    """Is fpk sitting on a harmonic k*f0? Returns k, else None.
    Two sources of harmonics matter and they are DIFFERENT:
      - the mean FIRING RATE (v3 checked this one)
      - the DRIVE frequency (v3 did not; a sharply theta-locked population rate
        puts harmonics of 6 Hz at 36, 42, 48, 54, 60 Hz -- squarely inside the
        gamma band, where they are picked up as 'gamma')."""
    if not f0 or f0 <= 0 or not np.isfinite(fpk):
        return None
    k = int(round(fpk / f0))
    if 1 <= k <= max_k and abs(fpk - k * f0) <= tol_bins * df:
        return k
    return None


def test_rhythm(spike_times_s, n_neurons, T_s, band=(30.0, 80.0),
                n_surr=200, method="poisson", dt_s=DT_S, seed=0,
                drive_hz=None, check_rate_harmonic=True, notch=True):
    """Observed prominence vs a surrogate null, PLUS harmonic screening.

    IMPORTANT: the surrogate null alone is NOT sufficient. It asks "is there
    non-random fast structure here?" -- and the harmonics of a sharply
    theta-locked firing pattern ARE non-random fast structure. The jitter
    surrogate destroys them, so they come back SIGNIFICANT. They are real, they
    are simply not a rhythm. A peak must clear THREE screens:
        (1) p < 0.05 against the surrogate null   [is it above chance?]
        (2) not at a band edge                    [is it a fit artifact?]
        (3) not a harmonic of the firing rate OR of the drive  [is it its own?]
    Measured example: the 320-cell net gives p = 0.010 with the peak at 54.9 Hz.
    The drive is 6 Hz. 9 x 6 = 54. That is not gamma."""
    rng = np.random.default_rng(seed)
    fs = 1.0 / dt_s

    rate = pop_rate(spike_times_s, n_neurons, T_s, dt_s)
    f, P = psd(rate, fs)
    mean_rate = len(spike_times_s) / (n_neurons * T_s)

    # If a drive frequency is known, NOTCH ITS COMB OUT before taking the max.
    # The un-notched statistic is blind on a driven network (see prominence_notched).
    # The surrogates are scored with the identical statistic, so the null is the
    # null OF THE STATISTIC ACTUALLY USED.
    use_notch = notch and (drive_hz is not None)
    stat = ((lambda ff, PP: prominence_notched(ff, PP, band, drive_hz))
            if use_notch else (lambda ff, PP: prominence(ff, PP, band)))
    obs, fpk, slope = stat(f, P)

    null = np.empty(n_surr)
    for i in range(n_surr):
        if method == "poisson":
            r_s = poisson_surrogate(rate, n_neurons, band, rng, dt_s)
        elif method == "jitter":
            st = jitter_surrogate(spike_times_s, band, T_s, rng)
            r_s = pop_rate(st, n_neurons, T_s, dt_s)
        else:
            raise ValueError(method)
        fs_, Ps_ = psd(r_s, fs)
        null[i], _, _ = stat(fs_, Ps_)

    p = (1.0 + np.sum(null >= obs)) / (1.0 + n_surr)
    z = (obs - null.mean()) / (null.std() + 1e-12)

    # ---- screens 2 and 3 ----
    df = f[1] - f[0]
    band_edge = (abs(fpk - band[0]) <= 1.5 * df) or (abs(fpk - band[1]) <= 1.5 * df)
    k_drive = harmonic_of(fpk, drive_hz, df) if drive_hz else None
    k_rate = harmonic_of(fpk, mean_rate, df) if check_rate_harmonic else None
    clean = (p < 0.05) and not band_edge and k_drive is None and k_rate is None

    reason = ("above chance, no band-edge, no harmonic" if clean
              else "; ".join(filter(None, [
                  None if p < 0.05 else f"p={p:.3f} not significant",
                  "peak at band edge" if band_edge else None,
                  f"peak = {k_drive}x drive ({drive_hz} Hz)" if k_drive else None,
                  f"peak = {k_rate}x firing rate ({mean_rate:.1f} Hz)" if k_rate else None,
              ])))

    return dict(obs_db=obs, fpk=fpk, slope=slope, mean_rate=float(mean_rate),
                notched=bool(use_notch),
                null_mean=float(null.mean()), null_sd=float(null.std()),
                null_p95=float(np.percentile(null, 95)),
                z=float(z), p=float(p),
                significant=bool(p < 0.05),          # passed the NULL only
                band_edge=bool(band_edge),
                k_drive=k_drive, k_rate=k_rate,
                is_rhythm=bool(clean),               # passed ALL THREE screens
                reason=reason, method=method, null=null)


def report(name, r):
    verdict = "SIGNIFICANT" if r["significant"] else "not significant"
    print(f"{name}")
    print(f"    observed prominence : {r['obs_db']:6.2f} dB @ {r['fpk']:.1f} Hz")
    print(f"    NULL (n={len(r['null'])}, {r['method']}) : "
          f"mean {r['null_mean']:5.2f} +/- {r['null_sd']:.2f} dB, "
          f"95th pct {r['null_p95']:5.2f} dB")
    print(f"    z = {r['z']:+5.2f}   p = {r['p']:.4f}   ->  {verdict}")
    print()


# ---------------------------------------------------------------- self-test
def _make_spikes(T_s, n_neurons, base_hz, mod_hz=None, mod_depth=0.0,
                 theta_hz=6.0, theta_depth=0.0, rng=None, dt_s=DT_S):
    """Inhomogeneous Poisson spikes with optional theta and/or fast modulation."""
    rng = rng or np.random.default_rng(0)
    t = np.arange(0, T_s, dt_s)
    lam = np.full_like(t, base_hz)
    if theta_depth:
        lam = lam * (1 + theta_depth * np.sin(2 * np.pi * theta_hz * t))
    if mod_hz and mod_depth:
        lam = lam * (1 + mod_depth * np.sin(2 * np.pi * mod_hz * t))
    lam = np.clip(lam, 0, None)
    counts = rng.poisson(lam * n_neurons * dt_s)
    idx = np.nonzero(counts)[0]
    st = np.repeat(t[idx], counts[idx])
    return st


if __name__ == "__main__":
    # T MUST match the analysis configuration. An earlier version of this block
    # used T = 2.5 s while artifact_demo.py used 21 s, so the two files reported
    # DIFFERENT nulls (9.66 dB vs 5.62 dB) for the same statistic. The null of
    # peak prominence depends on the number of Welch bins AND the number of Welch
    # segments; both depend on T. Never quote a null without its configuration.
    T, N = 21.0, 80
    N_FP, N_PW, NS = 40, 20, 300

    print("=" * 78)
    print("VALIDATION OF THE NULL TEST")
    print("=" * 78)
    print("A test is validated by its ERROR RATES OVER MANY DATASETS, never by one")
    print("draw. At alpha = 0.05 a correct test rejects a true null 5% of the time;")
    print("seeing that happen once is not a bug, it is the definition. (An earlier")
    print("version of this file declared itself broken on exactly such a draw.)")
    print()

    # ---- false-positive rate: data with NO gamma ----
    print(f"FALSE-POSITIVE RATE — {N_FP} independent datasets with NO gamma")
    print("  (constant-rate Poisson + 6 Hz theta; gamma content zero by construction)")
    for meth in ("jitter", "poisson"):
        fp, obs, nul = 0, [], []
        for k in range(N_FP):
            st = _make_spikes(T, N, base_hz=10.0, theta_depth=0.5,
                              rng=np.random.default_rng(900 + k))
            r = test_rhythm(st, N, T, band=(30, 80), n_surr=NS,
                            method=meth, seed=k)
            fp += r["significant"]; obs.append(r["obs_db"]); nul.append(r["null_mean"])
        print(f"    {meth:8s}: {fp:2d}/{N_FP} rejected = {100*fp/N_FP:4.1f}%  "
              f"(nominal 5%)   raw prominence {np.mean(obs):.2f} dB vs null "
              f"{np.mean(nul):.2f} dB")
    print()

    # ---- power: data WITH real gamma ----
    print(f"POWER — {N_PW} datasets with a genuine 40 Hz rhythm")
    for meth in ("jitter", "poisson"):
        hit = 0
        for k in range(N_PW):
            st = _make_spikes(T, N, base_hz=10.0, theta_depth=0.5,
                              mod_hz=40.0, mod_depth=0.6,
                              rng=np.random.default_rng(300 + k))
            r = test_rhythm(st, N, T, band=(30, 80), n_surr=NS,
                            method=meth, seed=k)
            hit += r["significant"]
        print(f"    {meth:8s}: {hit:2d}/{N_PW} detected = {100*hit/N_PW:3.0f}% power")
    print()

    # ---- the bias the test exists to correct ----
    vals = []
    for k in range(200):
        st = _make_spikes(T, N, base_hz=10.0, rng=np.random.default_rng(7000 + k))
        rate = pop_rate(st, N, T)
        f, P = psd(rate, 1.0 / DT_S)
        vals.append(prominence(f, P, (30, 80))[0])
    v = np.array(vals)
    print("=" * 78)
    print("WHY THE TEST IS NEEDED — the null of the raw statistic")
    print("=" * 78)
    print(f"  200 datasets with NO rhythm of any kind. An unbiased statistic")
    print(f"  would return ~0 dB. Peak prominence returns:")
    print(f"      mean {v.mean():.2f} dB | sd {v.std():.2f} | 95th {np.percentile(v,95):.2f} "
          f"| max {v.max():.2f}")
    print(f"  => raw prominence below ~{np.percentile(v,95):.1f} dB is not evidence of anything.")
    print()
    print("  CAUTION: passing this null is necessary and NOT sufficient. Harmonics of")
    print("  an imposed drive are non-random fast structure, so the surrogate calls")
    print("  them significant. Use test_rhythm(..., drive_hz=...) and read is_rhythm,")
    print("  not significant. See artifact_demo.py.")
    print("=" * 78)
