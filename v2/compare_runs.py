#!/usr/bin/env python3
"""
compare_runs.py -- do two runs report the same numbers?
Larry (Peilin) Zhong

WHY THIS EXISTS

run_all.sh gates on the SHA-256 of the source bytes. That is correct, and it
caught two machines silently executing different code. But it means a comment,
a docstring, or a corrected line of printed prose changes the fingerprint, and a
cross-machine log that verified every number in the paper no longer matches the
archive -- although every number in it is still right.

The alternatives are both bad: re-run an hour of simulation to certify a comment,
or write "only the comments changed" in the paper and ask the reader to trust it.

This script is the third option. It strips the things that are ALLOWED to differ
between two runs -- fingerprints, versions, wall-clock timings, log filenames --
and compares everything else, line by line. If two runs of two different archives
report identical numbers, that is not an opinion about the diff. It is the two
outputs, checked.

Usage
-----
  python3 compare_runs.py ../results/old.log ../results/new.log

Exit 0 if every reported number is identical, 1 otherwise.
"""

import pathlib
import re
import sys

# Lines that are EXPECTED to differ between two runs of the same science, and
# which therefore carry no scientific claim. Everything not matched here is
# compared exactly.
VOLATILE = [
    re.compile(r'sha256\[:12\]'),                 # source fingerprints
    re.compile(r'^\s*\S+\.py\s+[0-9a-f]{12}\s+OK'),
    re.compile(r'finished in \d+s'),              # wall clock
    re.compile(r'\[.*finished in.*exit \d+\]'),
    re.compile(r'^\s*cost\(s\)'),                 # header of the wall-clock column
    re.compile(r'results_\d+_\d+\.log'),          # log filename
    re.compile(r'^\s*(brian2|numpy|scipy|matplotlib|python)\s*[=|]'),
    re.compile(r'^\s*Python \d'),
    re.compile(r'brian2 \d+\.\d+'),
]

# The wall-clock column inside ca1_v2's results table is machine-dependent and
# sits between two numbers that are not. Blank it rather than dropping the row.
COST_COL = re.compile(r'(\s)\d+\.\d(\s+0\.000\s)')


# A number, or a verdict. These are the script's CLAIMS. Everything else printed
# is prose: the explanatory text a script writes around its results. Rewording
# prose is not a scientific change and must not be reported as one -- but a
# comparison that ignores prose entirely would also miss a script that quietly
# stopped printing a warning, so prose differences are reported, and do not fail.
NUMBER = re.compile(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?')
VERDICT = re.compile(
    r'\b(OK|GAMMA|THETA|RHYTHM|rejected|correctly rejected|correctly found|'
    r'CRITERION MET|CRITERION NOT MET|no gamma|FOUND|MISSED|significant|'
    r'usable|DESTROYS THE BAND|PASSES|WRONG|FAILED|ABORTING)\b')


def clean(path):
    out = []
    for line in pathlib.Path(path).read_text(errors="ignore").splitlines():
        if any(p.search(line) for p in VOLATILE):
            continue
        line = COST_COL.sub(r'\1[cost]\2', line)
        line = line.rstrip()
        if line:
            out.append(line)
    return out


def claims(lines):
    """Every number and every verdict, in order. This is what a run ASSERTS."""
    out = []
    for ln in lines:
        out.extend(NUMBER.findall(ln))
        out.extend(m.group(0) for m in VERDICT.finditer(ln))
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <old.log> <new.log>")
    a, b = clean(sys.argv[1]), clean(sys.argv[2])
    ca, cb = claims(a), claims(b)

    print("=" * 78)
    print("RUN COMPARISON")
    print("=" * 78)
    print(f"  old : {sys.argv[1]}")
    print(f"  new : {sys.argv[2]}")
    print(f"  claims (numbers + verdicts): {len(ca)} vs {len(cb)}\n")

    import difflib

    if ca != cb:
        d = [x for x in difflib.unified_diff(ca, cb, "old", "new", n=0, lineterm="")
             if x.startswith(("+", "-")) and not x.startswith(("+++", "---"))]
        print(f"  *** {len(d)} CLAIM(S) DIFFER. The change was NOT inert. ***\n")
        for x in d[:60]:
            print("   ", x)
        if len(d) > 60:
            print(f"    ... and {len(d) - 60} more")
        print()
        print("  A number or a verdict moved. Do not carry a cross-machine")
        print("  verification across this change. Re-run.")
        print("=" * 78)
        return 1

    print("  EVERY NUMBER AND EVERY VERDICT IS IDENTICAL.")
    print()
    print("  The two archives differ in their source fingerprints and assert exactly")
    print("  the same results. A cross-machine verification of one therefore carries")
    print("  to the other, and no re-run is needed to establish that.")

    prose = [x for x in difflib.unified_diff(a, b, "old", "new", n=0, lineterm="")
             if x.startswith(("+", "-")) and not x.startswith(("+++", "---"))]
    if prose:
        print()
        print(f"  {len(prose)} line(s) of PROSE were reworded. Listed, not counted:")
        print()
        for x in prose[:30]:
            print("   ", x)
        if len(prose) > 30:
            print(f"    ... and {len(prose) - 30} more")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
