# v2.1 — figures, and the second copy that went stale

**Copy the section below into the GitHub Release description. Zenodo will pull it into the record.**

---

## No number in this release is different from v2.0. This release exists so that the v2 preprint can cite a code DOI whose snapshot actually produces the figures the preprint prints.

The four gated source files are byte-for-byte identical to v2.0:

```
spectral_null.py     2e3b45bc1032
ca1_v2.py            b18a329f5027
artifact_demo.py     a6d20715722c
ping_scaling_test.py 346c8a9d255b
```

`run_all.sh` will still refuse to run if those hashes do not match, and the logs in `results/` are the same logs. **Nothing needs to be re-run.**

---

### What changed

**The figures were redrawn.** Figure 1 carried four inline text annotations and all four collided: with each other, with the dotted lines, and with the bars. Figure 2's `α = 5%` label sat inside the rightmost bar and could not be read. Figure 4B's legend was in the lower left while every other legend in the paper was above the axes.

That last one is the interesting one. The rule was already written down, in a comment in `make_figures.py`, and then broken three panels later. A rule enforced by a human reading a comment is not a rule. Legends are now placed by a function that cannot put them anywhere else, and no axes anywhere carry inline annotation: everything goes in the legend or in the caption.

**The figures now have one home.** `make_figures.py` used to write into `v2/`, and the figures were then copied by hand into `figures/`. So there were two copies. When the figures were redrawn, one of them was updated.

This repository has been shipping a `v2/` directory whose figures were several revisions behind the `figures/` directory sitting next to it, and nothing in the repository said so. That is precisely the failure this paper is about, reproduced in the repository of the paper about it. `make_figures.py` now writes to `figures/` and only to `figures/`. The duplicates are deleted.

**Every script now reports the versions it actually loaded** (brian2, numpy, scipy, matplotlib, python). `make_figures.py` was the one script that did not, so no log ever recorded a matplotlib version, so the matplotlib pin in `requirements-frozen.txt` had to be *guessed*. A guess in a pinned-version file is indistinguishable, on inspection, from a measurement. It is measured now.

**Dependencies are split in two.** `requirements.txt` gives floors, which is what you need to run the code. `requirements-frozen.txt` gives exact versions, which is what you need to reproduce the logged numbers to the last digit. A `>=` floor pins nothing, and a reproducibility claim cannot rest on one.

**Removed from the repository:** `fix_tables.py` and `progress.py`. Neither is imported by anything; the first is manuscript tooling and belongs with the manuscript, the second was a scratch file.

---

### The scientific result, unchanged

```
T_cycle = 16.8 + 0.760 · tau_GABA    (ms)

slope +0.760 ± 0.113 | R² = 0.884 | p = 0.0005
```

The 320-neuron network has emergent PING gamma. The 80-neuron network does not (p = 0.75). Network size, not reduction, is the switch. Criterion fixed in advance, seeds never previously used, reproduced to the last digit on two machines with different Python, numpy and hardware.

---

### Citing

| | DOI |
|---|---|
| v2 preprint | *(fill in after Zenodo mints it)* |
| v2 code | *(this release)* |
| v1 preprint | 10.5281/zenodo.21272959 |
| v1 code | 10.5281/zenodo.21273211 |

Code MIT. Manuscript CC-BY-4.0, on Zenodo, and not in this repository, for the reason given above.
