# CA1 microcircuit simplification: emergent gamma, and the artifacts that hide it

**Peilin (Larry) Zhong** · Shady Side Academy, Pittsburgh, PA

---

> ## ⚠️ Version 1 of this work reached a conclusion that Version 2 reverses.
>
> **v1 concluded that emergent gamma is absent from reduced CA1 models. That conclusion is wrong**, and it is wrong for two independent reasons: v1's "networks" contained essentially no synapses, and the spectral statistic it used is blind to gamma in a theta-driven network.
>
> **v1 is preserved here, unmodified, on purpose.** The corrigendum in the v2 paper cannot be independently checked without it. It is not deleted, and it should not be.
>
> If you are here for the science, read **v2**. If you are here to check the corrigendum, read **v1** as well.

---

## What v2 found

Two things, and they are inseparable.

**1. The standard analysis is broken for driven networks.** In these models theta is *imposed* as a drive rather than emergent. If the population locks tightly enough to that drive, the drive's harmonic comb enters the gamma band. It does not merely add a spurious peak, it *becomes* the peak:

| | |
|---|---|
| Peak prominence on data with **no rhythm at all** | 5.62 ± 0.61 dB, not 0 |
| Gamma-free spike trains at σ ≤ 12 ms locking | 12–37 dB of "gamma", and a **surrogate null test calls them significant in 100% of datasets** |
| A **real** 40 / 55 / 68 Hz rhythm under tight locking | the raw statistic reports its peak at **30.0 Hz in every case** |

The last row is the important one. The comb does not only create false positives. **It masks real rhythms.** The statistic's output is nearly uncorrelated with the gamma content it claims to measure.

**2. Once the analysis is repaired, emergent gamma is there.** Excise the drive's harmonic comb before taking the maximum. Then, in a network with an excitatory–inhibitory loop that actually exists:

$$T_{\text{cycle}} = 16.8 + 0.760\,\tau_{\text{GABA}} \quad \text{(ms)}$$

slope +0.760 ± 0.113, R² = 0.884, **p = 0.0005**, on a criterion fixed in advance and seeds never previously used. That is PING. The 16.8 ms intercept is the fixed part of the loop delay.

**The 320-neuron network has it. The 80-neuron network does not. Network size, not reduction, is the switch.**

---

## The one idea worth taking away

Nine checks are listed in §4 of the paper. **Eight of them can only reject.** They remove candidate explanations, and they remove a great deal: each artifact is caught by exactly one of them, and each artifact survives the check that catches the previous one.

**No amount of screening can turn a surviving peak into a rhythm.** Only the ninth can, and it is not a statistic. It is an experiment: perturb the mechanism you are naming, and see whether the peak responds the way that mechanism requires.

In this study, four screens and a comb notch left a significant peak standing in two conditions. The ninth check killed it in one and confirmed it in the other.

---

## Reproducing it

```bash
cd v2
pip install brian2 numpy scipy matplotlib
bash run_all.sh          # ~1.5–2 h total
```

`run_all.sh` **refuses to run** if the source hashes do not match. That gate is not ceremony. Two machines once produced *"gamma 8.51 dB, verdict GAMMA"* and *"gamma 3.17 dB, verdict no gamma"* for what was believed to be the same experiment, because they were executing different files, and nothing in the output said so. With the gate in place, two machines with different Python (3.13 / 3.12), numpy (2.5 / 2.2) and hardware reproduced every table **to the last digit**.

Raw logs from both machines are in `results/`.

| file | what it does | runtime |
|---|---|---|
| `v2/spectral_null.py` | the statistic, the comb notch, the surrogates, and their validation (false-positive rate, power) | ~2 min |
| `v2/artifact_demo.py` | every artifact, on synthetic spike trains, **with no network at all** | ~4 min |
| `v2/ca1_v2.py` | the corrected CA1 microcircuit | ~15–25 min |
| `v2/ping_scaling_test.py` | **the pre-registered PING test** — the only check that confirms | ~50–90 min |
| `v2/make_figures.py` | the four figures | ~1 min |

---

## What was wrong in v1

All three are reproducible from `v1/CA1-simplification-v3.py`.

**C1 — the networks had no synapses.** One connection probability (2%) was applied to *every* pathway. But 2% is a fact about *pyramidal→pyramidal* connectivity in CA1; interneuron→pyramidal connectivity is an order of magnitude higher. Expected inhibitory→excitatory synapses in the 10-neuron nets: 2 × 8 × 0.02 = **0.32**. Realised, across v1's own seeds 100–104:

```
inhibitory -> excitatory synapses : [0, 0, 0, 0, 0]
TOTAL synapses in the whole network: [1, 0, 0, 3, 0]
```

It was ten uncoupled neurons sharing a 6 Hz drive. The sparsification arm removed connections from a network that had none.

**C2 — the interneurons never fired.** The hybrid network, whose entire purpose was to test whether detailed inhibitory kinetics recover gamma, drove its Hodgkin–Huxley interneurons at I = 1.0 µA/cm². Their rheobase is 6–7. They emitted **zero spikes**.

**C3 — the statistic had no null, and was the wrong statistic.** v1 reported prominence against an implicit null of 0 dB. The null is ~5.6 dB. And on a driven network the statistic reports the drive's harmonics rather than the gamma: v1's gamma peak sat at exactly **30.0 Hz in every single condition**, and 30 = 5 × 6 Hz.

**What survives from v1:** the single-neuron HH→LIF cost comparison, and the decision to work with prominence above a fitted aperiodic background rather than raw band power. v2 repairs that statistic rather than abandoning the idea behind it.

---

## Citing

| | DOI |
|---|---|
| **v2 preprint** | *(pending)* |
| **v2 code** | *(pending)* |
| v1 preprint | [10.5281/zenodo.21272959](https://doi.org/10.5281/zenodo.21272959) · all versions: `10.5281/zenodo.21272958` |
| v1 code | [10.5281/zenodo.21273211](https://doi.org/10.5281/zenodo.21273211) · all versions: `10.5281/zenodo.21265445` |

---

## Layout

```
v2/          the corrected code. Start here.
v1/          v1 source, unmodified. Kept so the corrigendum can be checked.
results/     raw run logs from both machines. The reproducibility claim lives here.
figures/     the four figures (pdf + png), regenerable by v2/make_figures.py
```

**The manuscript is not in this repository.** It lives on Zenodo, where it has a
DOI, a version history and an archival guarantee. A second copy here would be a
copy that goes stale the first time the paper is revised, and it would go stale
silently. That is the exact failure mode this paper is about, and it would be a
poor place to reproduce it.

MIT licence for the code. Manuscript CC-BY-4.0 on Zenodo.
