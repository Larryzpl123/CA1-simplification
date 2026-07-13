# Version 1 source — DO NOT DELETE, DO NOT "FIX"

`CA1-simplification-v3.py` is the source code of Version 1, **exactly as it was
published**. ("v3" is the third code draft; there is only one prior preprint.)

Its central conclusion is reversed by v2. It is kept here, unmodified, because
**the corrigendum cannot be independently checked without it.**

Anyone should be able to run this file and see for themselves:

```python
conn = (rng.random((N, N)) < p_conn)      # p_conn = 0.02, applied to EVERY pathway
```

For the 10-neuron networks, across v1's own seeds 100-104:

    inhibitory -> excitatory synapses : [0, 0, 0, 0, 0]
    TOTAL synapses in the whole net   : [1, 0, 0, 3, 0]

Three of five seeds produced a network with zero synapses of any kind. It was
ten uncoupled neurons sharing a 6 Hz drive.

Do not repair this file. Its value is that it is wrong in a checkable way.
