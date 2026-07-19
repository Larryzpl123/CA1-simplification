# The run logs, and why one of them has different fingerprints

Every log in this directory prints, at the top, the SHA-256 fingerprint of every
source file that produced it. Three of the four match the code in `v2/`. One does
not, deliberately, and this file explains why, because a log whose hashes do not
match the archive is normally a sign of rot and here it is the opposite.

## The four logs

| log | machine | archive | what it ran |
|---|---|---|---|
| `larry_v23_20260714_122129.log` | Python 3.13.12, numpy 2.5.1 | **current** | everything |
| `larry_v22_20260714_005035.log` | Python 3.13.12, numpy 2.5.1 | previous | everything |
| `harry_v22_20260714_005128.log` | Python 3.12.1, numpy 2.2.4 | previous | `v1_diagnosis`, `artifact_demo`, `ca1_v2` |
| `harry_ping_20260714_115650.log` | Python 3.12.1, numpy 2.2.4 | previous | `ping_scaling_test` |

The "previous" archive differs from the current one only in comments, docstrings,
and the wording of some printed prose, plus one printed caveat in `ca1_v2.py` that
used to quote two stale numbers and now computes them.

## Why the older logs are kept

The paper claims that two machines, with different Python and different numpy,
report identical numbers. That claim rests on a cross-machine comparison, and a
cross-machine comparison requires two machines to have run the same code.

Re-running an hour of simulation on a second machine to certify a comment change
is not a good use of anyone's afternoon, and asserting "only the comments moved"
in the paper is exactly the kind of unchecked claim this work is about. So the
chain is made explicit and each link is checkable:

    harry_v22 + harry_ping  ==  larry_v22      (two machines, same archive)
    larry_v22               ==  larry_v23      (one machine, two archives)
    -------------------------------------------------------------------
    => every number in the current archive is verified on two machines

The second link is not an assertion. Run it:

    cd ../v2
    python3 compare_runs.py ../results/larry_v22_20260714_005035.log \
                            ../results/larry_v23_20260714_122129.log

`compare_runs.py` strips what is allowed to differ between two runs of the same
science — fingerprints, versions, wall-clock timings, log filenames — and compares
every remaining number and every verdict. It reports 903 claims, all identical,
and lists the four lines of prose that were reworded. It exits non-zero if a
number or a verdict moves; that behaviour is tested by feeding it a log with one
digit changed.

## Version 4: the current gated-code log, and one non-gated log

`larry_v4_20260717_163239.log` is the current run of the five gated scripts. Its
fingerprints (`ca1_v2` 817b29fbb9b6, `ping_scaling_test` b2f43bf04529) are the
Version 4 hashes. The change from the earlier gated hashes is comments, a
docstring, and printed prose only: the intercept is no longer described as a loop
delay, and `ca1_v2`'s false "size is the only thing that varies" comment is gone.
`compare_runs.py` certifies larry_v23 against larry_v4 as every number and every
verdict identical, three prose tokens aside (the withdrawn intercept sentence).

`jobB_20260718_123645.log` is different in kind. It is produced by
`v2/job_b_noise_matched.py`, which is **not in the hash gate**, on purpose:
that script computes no gated result. It imports `ca1_v2.build_and_run` and
`ping_scaling_test.test_rhythm` unchanged and calls them with different connection
probabilities, to ask whether the 80 + 20 network's failure to show PING is about
the cell count or the input fluctuations. Its 320 + 80 control reproduces the
gated `larry_v4` fit to the last digit (16.8 + 0.760 τ, p = 0.0005), which is what
lets the new 80 + 20 result be trusted: the gated code is unchanged, only the
probabilities differ. The result is Figure 6 and the close of §3.8.

## Older logs are not kept

Logs from archives that produced *different numbers* are not here. They are not
history, they are noise: a reader who finds a log in an archive is entitled to
assume it certifies that archive, and eight superseded logs sitting next to the
code is the failure this paper documents.
