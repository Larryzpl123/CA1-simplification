# Simplification Strategies for Biologically Constrained Hippocampal CA1 Microcircuit Models

**How far reduction preserves firing and driven theta but not emergent gamma.**

Peilin (Larry) Zhong — Shady Side Academy, Pittsburgh, PA, USA

This repository contains the full simulation and analysis code for a study of how
far a biologically constrained CA1 microcircuit can be simplified before its
dynamics degrade. Hodgkin–Huxley (HH) and leaky integrate-and-fire (LIF) neuron
models are compared at four scales (single neuron, 10-neuron network, 100-neuron
scaled network, and a hybrid LIF/HH network) under three reduction strategies:
morphological reduction, ion-channel abstraction, and connectivity sparsification.

## Key findings

- Firing rates and a driven 6 Hz theta component are preserved across all three simplifications.
- Channel abstraction (HH to LIF) reduces single-neuron computational cost roughly twentyfold.
- No genuine emergent gamma rhythm appears at any scale; apparent gamma peaks are firing-rate harmonics or band-edge artifacts.

## Contents

| File | Description |
|------|-------------|
| `ca1_simplification.py` | Full simulation + analysis (all 8 model configurations, spectral metrics, figures) |
| `results_v3.tsv` | Output metrics table |
| `figure1_voltage_traces.png` … `figure6_hybrid_network_power.png` | Generated figures |
| `requirements.txt` | Python dependencies |

## Reproducing the results

```bash
pip install -r requirements.txt
python ca1_simplification.py
```

Runtime is a few minutes on a laptop. The script checkpoints each model block, so
it can be re-run without repeating completed work. All models use a common
fixed-step forward-Euler integrator (dt = 0.01 ms, 1000 ms, 5 seeded runs) so that
timing comparisons reflect model complexity rather than solver choice. Spectral
metrics use a Welch PSD with a fitted 1/f background; reported peak prominence and
peak frequency indicate whether a genuine rhythm is present.

## Citation

If you use this code, please cite the preprint:

> Zhong, P. (2026). *Simplification Strategies for Biologically Constrained
> Hippocampal CA1 Microcircuit Models: How Far Reduction Preserves Firing and
> Driven Theta but Not Emergent Gamma* (v1.0). Zenodo.
> https://doi.org/10.5281/zenodo.21272959

Archived code (this repository): https://doi.org/10.5281/zenodo.21265446

## License

Code released under the MIT License (see `LICENSE`).
