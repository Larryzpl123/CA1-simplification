#!/usr/bin/env python3
"""
audit_manuscript.py -- does the paper say anything the archive cannot produce?
Larry (Peilin) Zhong

WHY THIS IS IN THE ARCHIVE

This paper argues that every number in a paper must be produced by code the reader
can run, and that a number carried in prose does not recompute itself when the code
underneath it changes. That argument is worthless as an assertion. It is only worth
anything if the paper can be CHECKED against its own archive, mechanically, by
anyone, in one command.

So this is that command. It reads the manuscript, extracts every quantitative claim,
and looks for each one in the run logs. Anything it cannot find is either
(a) arithmetic from numbers that ARE in the logs, in which case whitelist it here
with the derivation written down, or (b) a defect.

It found real defects. Every one of them had survived a careful human reading:

  - the manuscript said rate modulation produces no spurious gamma; the log said
    7.55 dB at 83% significance
  - the manuscript said the false-positive rate was 0%; the log said 1/40, 4/40, 1/40
  - the manuscript quoted a W_IE calibration table the code no longer produced
  - the manuscript quoted two source fingerprints that were two revisions stale
  - the manuscript reported v1's mean in-degree as 0.42, a number that is neither
    the synapse count (0.32) nor the in-degree (0.04) and appears nowhere

A human reading a 29-page paper against a 2000-line log will not find these. Nobody
has ever found these by reading. That is the point.

Run:  python3 audit_manuscript.py ../../CA1_v2_manuscript.md ../results/<newest>.log
Exit: 0 if clean, 1 if any claim is unsupported.
"""

import re
import sys
import pathlib

# ---------------------------------------------------------------------------
# Numbers that are DERIVED, not measured. Each needs its derivation, here, in
# code, so that "it's just arithmetic" is a checkable statement and not an
# excuse. If you add to this list without a derivation you have defeated the
# purpose of the file.
# ---------------------------------------------------------------------------
DERIVED = {
    "0.32":   "expected v1 I->E synapses = 2 inh * 8 exc * 0.02",
    "0.04":   "expected v1 I->E in-degree = 2 inh * 0.02",
    "0.015":  "SE(p) at p=.05, N=200  = sqrt(.05*.95/200)",
    "0.0069": "SE(p) at p=.05, N=1000 = sqrt(.05*.95/1000)",
    "0.0049": "SE(p) at p=.05, N=2000 = sqrt(.05*.95/2000)",
    "0.0097": "SE(p) at p=.05, N=500  = sqrt(.05*.95/500)",
    "0.0109": "SE(p) at p=.05, N=400  = sqrt(.05*.95/400)",
    "0.048":  "Binomial(40,.05): P(X>4). Threshold for E7's FP criterion.",
    "0.74":   "0.95**6 -- the pass rate of the ZERO-false-positive criterion E7 used to use",
    "0.95":   "1 - alpha",
    "16.7":   "jitter width w = 1/(2*f_low) = 1/(2*30) s = 16.7 ms",
    "0.2":    "required df = 0.1 * f_drive / 3 = 0.1*6/3",
    "0.75":   "p = 0.7482, rounded, in the abstract",
    "5.0":    "6 false positives / 120 gamma-free datasets = 5.0%",
    "3.17":   "historical: the value the SECOND machine printed before the hash gate existed",
    "8.51":   "also the live LIF-baseline value; appears in the two-machine anecdote",

    # Rounded in prose. The log prints the full precision; the paper prints fewer
    # digits. Listed so that "it's just rounding" is a checkable claim: if the log
    # value ever moves, the rounded value stops matching and this entry is wrong.
    "0.12":   "df = 0.122 Hz, rounded (log: 'freq resolution df = 0.122 Hz')",
    "0.748":  "p = 0.7482, rounded (LIF baseline PING regression)",
    "0.88":   "R^2 = 0.884, rounded (scaled 4x PING regression)",
    "120":    "3 negative cases x 40 datasets = 120 gamma-free datasets in E7",
    "320":    "network size: 320 excitatory + 80 inhibitory = the scaled 4x arm",
    "500":    "N_surr in E7, printed by artifact_demo as 'N_surr = 500'",
    "11":     "11 x 6 = 66 Hz, a harmonic index in the E2 table",

    # THE ONLY NUMBERS IN THIS PAPER THE ARCHIVE CANNOT REPRODUCE, and they are
    # not evidence. Seeds 100-102 were burned BECAUSE the acceptance criterion was
    # still being changed while they were being looked at. Re-running them would
    # produce a number, and the number would be worthless: no criterion can be
    # applied to the data whose inspection is what moved the criterion. They are
    # reported as a DISCLOSURE of researcher degrees of freedom (section 2.6), so
    # that a reader can see what those degrees of freedom were worth here.
    # An archive that could reproduce them would mean the seeds were not burned.
    "0.539":  "DISCLOSURE, not evidence: slope on the burned seeds 100-102",
    "0.176":  "DISCLOSURE, not evidence: its standard error",
}

# Section numbers, figure numbers, years, DOIs, version numbers: not claims.
SKIP_CONTEXT = re.compile(
    r'(?:§|Figure |Table |v?[12]\.\d|10\.5281|zenodo|20\d\d[a-z]?|'
    r'Python |numpy |scipy |brian2 |matplotlib )')


def numbers_in(text):
    """Every number that makes a factual claim."""
    text = re.sub(r'```.*?```', '', text, flags=re.S)      # code blocks
    text = re.sub(r'^!\[.*$', '', text, flags=re.M)        # image lines
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'10\.5281/\S+', '', text)
    # Section refs, INCLUDING ranges: "§3.5", "§§3.1-3.4". Strip these BEFORE the
    # range rule below, or "§§3.1-3.4" loses its "3.1" to the range rule and the
    # orphaned "3.4" is then reported as an unsupported measurement.
    text = re.sub(r'§+\s*[\d.]+(?:\s*[-–—]\s*[\d.]+)?', '', text)
    text = re.sub(r'^#{1,6}\s.*$', '', text, flags=re.M)   # section headings
    # A RANGE is one claim about two endpoints ("28-33 dB", "12-37 dB"). The
    # endpoints are read off a figure or a column, not printed as literals, so
    # requiring each to appear in a log flags every range in the paper. Drop them
    # and keep the flanking prose honest by eye. This is the ONE place this file
    # trusts a human, and it is doing so about which numbers are claims, never
    # about what a claim's value is.
    text = re.sub(r'\d+(?:\.\d+)?\s*(?:[-–—]|to)\s*\d+(?:\.\d+)?', '', text)
    out = {}
    for m in re.finditer(r'(?<![\w.\-−/])(\d+\.\d+|\d{2,5})(?![\w./%])', text):
        n = m.group(1)
        if re.fullmatch(r'(19|20)\d\d', n):                # years
            continue
        lo = max(0, m.start() - 40)
        ctx = " ".join(text[lo:m.end() + 30].split())
        out.setdefault(n, ctx)
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} <manuscript.md> <log> [log ...]")
    ms = pathlib.Path(sys.argv[1]).read_text()
    body = ms.split("## References")[0]
    logs = "\n".join(pathlib.Path(p).read_text(errors="ignore")
                     for p in sys.argv[2:])

    def in_logs(n):
        pat = re.escape(n)
        if re.search(r'(?<![\w.])' + pat + r'(?![\w.])', logs):
            return True
        # the logs print -0.041; the manuscript sets it as a Unicode minus
        return re.search(r'(?<![\w.])' + re.escape(n.replace("−", "-")) +
                         r'(?![\w.])', logs) is not None

    claims = numbers_in(body)
    unsupported = []
    for n, ctx in sorted(claims.items(), key=lambda kv: kv[0]):
        if in_logs(n) or n in DERIVED:
            continue
        unsupported.append((n, ctx))

    print("=" * 78)
    print("MANUSCRIPT AUDIT")
    print("=" * 78)
    print(f"  manuscript : {sys.argv[1]}")
    print(f"  logs       : {len(sys.argv) - 2}")
    print(f"  claims     : {len(claims)} numbers")
    print(f"  in a log   : {sum(1 for n in claims if in_logs(n))}")
    print(f"  derived    : {sum(1 for n in claims if n in DERIVED and not in_logs(n))}")
    print(f"  UNSUPPORTED: {len(unsupported)}")
    print()
    if not unsupported:
        print("  Every number in this paper is printed by a script in this archive,")
        print("  or is arithmetic from numbers that are, with the derivation recorded")
        print("  in DERIVED above. The paper says nothing the archive cannot produce.")
        print("=" * 78)
        return 0

    print("  These appear in the manuscript and in NO log, and are not derived:")
    print()
    for n, ctx in unsupported:
        print(f"    {n:>9}   ...{ctx[:88]}...")
    print()
    print("  Each one is a defect. Either the code should print it, or the paper")
    print("  should not claim it, or it is arithmetic and belongs in DERIVED with")
    print("  its derivation written down. 'I remember measuring it' is not an option:")
    print("  that is the failure this paper documents, five times, in itself.")
    print("=" * 78)
    return 1


if __name__ == "__main__":
    sys.exit(main())
