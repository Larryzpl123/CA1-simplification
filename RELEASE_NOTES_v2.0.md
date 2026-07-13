# v2.0 — Version 1's central conclusion is reversed

**Copy the section below into the GitHub Release description. Zenodo will pull it into the record.**

---

## Version 1 concluded that emergent gamma is absent from reduced CA1 models. It is not absent. This release corrects that.

v1 is preserved unmodified in `v1/`, and its DOI still resolves. It is not deleted, and it should not be: the corrigendum cannot be checked without it.

---

### What was wrong

Three of v1's arms did not test what they were described as testing. All three are reproducible from `v1/CA1-simplification-v3.py`.

**The networks had no synapses.** One connection probability (2%) was applied to *every* pathway. But 2% is a fact about *pyramidal→pyramidal* connectivity in CA1; interneuron→pyramidal connectivity is an order of magnitude higher. Expected inhibitory→excitatory synapses in the 10-neuron nets: `2 × 8 × 0.02 = 0.32`. Realised, across v1's own seeds 100–104:

```
inhibitory -> excitatory synapses : [0, 0, 0, 0, 0]
TOTAL synapses in the whole network: [1, 0, 0, 3, 0]
```

Three of five seeds produced a network with zero synapses of any kind. It was ten uncoupled neurons sharing a 6 Hz drive. The sparsification arm removed connections from a network that had none.

**The interneurons never fired.** The hybrid network, whose entire purpose was to test whether detailed inhibitory kinetics recover gamma, drove its Hodgkin–Huxley interneurons at I = 1.0 µA/cm². Their rheobase is 6–7. They emitted zero spikes.

**The statistic was the wrong statistic.** v1 reported peak prominence against an implicit null of 0 dB. The null is ~5.6 dB. Worse, on a theta-driven network the statistic is *blind*: v1's gamma peak sat at exactly **30.0 Hz in every single condition**, and 30 = 5 × 6 Hz. It was reporting the drive's fifth harmonic.

---

### What v2 found

**1. The standard analysis is broken for driven networks, in both directions.**

In these models theta is *imposed* as a drive. If the population locks tightly enough to it, the drive's harmonic comb enters the gamma band. It does not merely add a spurious peak, it *becomes* the peak.

| measured on synthetic spike trains, with **no network at all** | |
|---|---|
| prominence on data with **no rhythm of any kind** | 5.62 ± 0.61 dB, not 0 |
| gamma-free trains at σ ≤ 12 ms phase-locking | 12–37 dB of "gamma", and a **surrogate null test calls them significant in 100% of datasets** |
| a **real** 40 / 55 / 68 Hz rhythm under tight locking | the raw statistic reports its peak at **30.0 Hz in every case** |

The last row is the one that matters. **The comb does not only create false positives. It masks real rhythms.** The statistic's output is nearly uncorrelated with the gamma content it claims to measure. A surrogate null is necessary and **not sufficient**.

**2. Once the analysis is repaired, emergent gamma is there.**

Excise the drive's harmonic comb before taking the maximum (validated 6/6 on ground truth: 0% false positives including the worst case, 0% false negatives, rhythms recovered to within 0.1 Hz). Then, in a network with an excitatory–inhibitory loop that actually exists:

```
T_cycle = 16.8 + 0.760 · τ_GABA    (ms)

slope +0.760 ± 0.113 | R² = 0.884 | p = 0.0005
```

That is PING. The 16.8 ms intercept is the fixed part of the loop delay (AMPA, membrane, refractory), which is why the *frequency* moves only 2× while τ_GABA moves 12×.

**The 320-neuron network has it. The 80-neuron network does not (p = 0.75). Network size, not reduction, is the switch.**

---

### On the pre-registration

The acceptance criterion for that test was **fixed in advance** and run on **seeds never previously used (200–204)**. It had to be.

Two earlier versions of the criterion each bolted an extra hard threshold onto the regression — first a 2× ratio on frequency, then a strict monotonicity requirement — and each rejected the same result. Neither threshold is a prediction of PING; both were added without derivation, **in a study whose entire subject is thresholds added without derivation.**

Changing an acceptance criterion twice, each time toward acceptance, is p-hacking however good the reasons sound. The only remedy was to fix the criterion before looking and test it on untouched data. **The effect grew rather than shrank** (p = 0.022 → 0.0005), which is what a real effect does.

That disclosure is in the paper, in the Methods, not in a footnote.

---

### Reproducibility

```bash
cd v2
pip install -r ../requirements.txt      # to run
pip install -r ../requirements-frozen.txt   # to reproduce the logged numbers
bash run_all.sh
```

`run_all.sh` **refuses to run if the source hashes do not match.** That gate is not ceremony. Two machines once produced *"gamma 8.51 dB, verdict GAMMA"* and *"gamma 3.17 dB, verdict no gamma"* for what was believed to be the same experiment, because they were executing different files, and nothing in the output said so.

With the gate in place, two machines — Python 3.13.12 / 3.12.1, numpy 2.5.1 / 2.2.4, different hardware — reproduced every table **to the last digit**, including the PING regression. Both raw logs are in `results/`.

The two numpy versions differ and the results did not. That is a stronger statement than pinning alone would have given, and it is only checkable because every run prints the versions it actually loaded rather than the ones it assumes.

---

### The one idea worth taking away

Nine checks are listed in §4 of the paper. **Eight of them can only reject.** Each artifact above is caught by exactly one of them, and each artifact survives the check that catches the previous one.

**No amount of screening can turn a surviving peak into a rhythm.** Only the ninth can, and it is not a statistic. It is an experiment: perturb the mechanism you are naming, and see whether the peak responds the way that mechanism requires.

Four screens and a comb notch left a significant peak standing in two conditions. The ninth check killed it in one and confirmed it in the other.

---

### Citing

| | DOI |
|---|---|
| v2 preprint | *(fill in after Zenodo mints it)* |
| v2 code | *(this release)* |
| v1 preprint | [10.5281/zenodo.21272959](https://doi.org/10.5281/zenodo.21272959) |
| v1 code | [10.5281/zenodo.21273211](https://doi.org/10.5281/zenodo.21273211) |

Code MIT. Manuscript CC-BY-4.0, on Zenodo (not in this repository — a second copy here would go stale silently the first time the paper is revised, which is the exact failure this work is about).
