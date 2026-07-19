#!/bin/bash
# run_all.sh — run the whole v2 battery, in order, with a hash gate.
# Larry (Peilin) Zhong
#
#   bash run_all.sh
#
# Everything is teed to results_<timestamp>.log. Paste that file back.
#
# The hash gate is not ceremony. Two machines once produced "gamma 8.51 dB,
# verdict GAMMA" and "gamma 3.17 dB, verdict no gamma" for what was believed to
# be the same experiment; they were running different files, and nothing in the
# output said so. This script refuses to run if the files are not the expected
# ones.

set -uo pipefail
cd "$(dirname "$0")" || exit 1

PY="${PYTHON:-python3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="results_${STAMP}.log"

# ---- expected fingerprints (sha256, first 12 hex chars) ----------------------
declare -a FILES=(spectral_null.py ca1_v2.py artifact_demo.py ping_scaling_test.py v1_diagnosis.py)
declare -a WANT=(5a50478f81c8 817b29fbb9b6 3a4732283381 b2f43bf04529 568c312de2b8)

sha12() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -c1-12
    else sha256sum "$1" | cut -c1-12; fi
}

echo "==============================================================================" | tee "$LOG"
echo "HASH GATE" | tee -a "$LOG"
echo "==============================================================================" | tee -a "$LOG"
FAIL=0
for i in "${!FILES[@]}"; do
    f="${FILES[$i]}"; want="${WANT[$i]}"
    if [ ! -f "$f" ]; then
        printf "  %-22s MISSING\n" "$f" | tee -a "$LOG"; FAIL=1; continue
    fi
    got="$(sha12 "$f")"
    if [ "$got" = "$want" ]; then
        printf "  %-22s %s  OK\n" "$f" "$got" | tee -a "$LOG"
    else
        printf "  %-22s %s  MISMATCH (expected %s)\n" "$f" "$got" "$want" | tee -a "$LOG"
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "" | tee -a "$LOG"
    echo "ABORTING. The files are not the expected ones, so any numbers this run" | tee -a "$LOG"
    echo "produced would not be comparable to anyone else's. Re-fetch the code." | tee -a "$LOG"
    echo "(If you edited a file on purpose, update WANT[] above and say so.)" | tee -a "$LOG"
    exit 1
fi

echo "" | tee -a "$LOG"
echo "  $($PY --version 2>&1)" | tee -a "$LOG"
echo "  $($PY -c 'import brian2,numpy,scipy;print("brian2",brian2.__version__,"| numpy",numpy.__version__,"| scipy",scipy.__version__)' 2>&1)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ---- the battery, in dependency order ---------------------------------------
#
# PIPESTATUS[0], not $?. The python is piped through grep and tee, and a pipeline's
# exit status is that of its LAST command -- tee, which always succeeds. Without
# this, every `raise SystemExit(...)` in the battery is swallowed and a failed
# validation is followed by a clean "DONE".
run () {
    echo "" | tee -a "$LOG"
    echo "##############################################################################" | tee -a "$LOG"
    echo "## $1" | tee -a "$LOG"
    echo "## $2" | tee -a "$LOG"
    echo "##############################################################################" | tee -a "$LOG"
    t0=$(date +%s)
    # --line-buffered is not optional. Without it grep block-buffers into the pipe
    # and the script appears to hang for its entire runtime: the per-tau progress
    # lines, flushed by Python, sit in grep's 4 KB buffer until the process exits.
    # A long-running job that prints nothing is indistinguishable from a hung one.
    "$PY" -u "$1" 2>&1 | grep --line-buffered -v -i "^WARNING\|INFO *brian2" | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    echo "   [${1} finished in $(( $(date +%s) - t0 ))s, exit ${rc}]" | tee -a "$LOG"
    if [ "$rc" -ne 0 ]; then
        echo "" | tee -a "$LOG"
        echo "##############################################################################" | tee -a "$LOG"
        echo "## ${1} FAILED (exit ${rc}). ABORTING." | tee -a "$LOG"
        echo "## Nothing below this line was computed. Do not report anything above it" | tee -a "$LOG"
        echo "## either: a battery that continues past a failed validation is a battery" | tee -a "$LOG"
        echo "## whose output you cannot use." | tee -a "$LOG"
        echo "##############################################################################" | tee -a "$LOG"
        exit "$rc"
    fi
}

# The hash gate above ALWAYS runs on all four files, whatever ONLY is set to. So a
# partial re-run still certifies that the whole archive is the archive. What ONLY
# skips is recomputation, never verification.
#
# ONLY=ping     -> just the PING test.
# ONLY=changed  -> just artifact_demo.py and ca1_v2.py. Correct to use when those
#                  are the only two files whose hashes have moved since the last
#                  full run, which the gate above has just proved: the other two
#                  are byte-identical, so re-running them can only reproduce a log
#                  that already exists. Cross-machine verification of an unchanged
#                  file does not expire.
if [ "${ONLY:-all}" = "ping" ]; then
    run ping_scaling_test.py "PING scaling: period vs tau_GABA regression (~30-40 min)"
elif [ "${ONLY:-all}" = "diag" ]; then
    run v1_diagnosis.py      "C1, C2 and the two Methods numbers, made executable (~2 min)"
elif [ "${ONLY:-all}" = "changed" ]; then
    run v1_diagnosis.py      "C1, C2 and the two Methods numbers, made executable (~2 min)"
    run artifact_demo.py     "the artifacts, on synthetic data with NO network at all"
    run ca1_v2.py            "the corrected CA1 microcircuit (~15-25 min)"
else
    run spectral_null.py     "validate the null test itself: false-positive rate and power"
    run v1_diagnosis.py      "C1, C2 and the two Methods numbers, made executable (~2 min)"
    run artifact_demo.py     "the artifacts, on synthetic data with NO network at all"
    run ca1_v2.py            "the corrected CA1 microcircuit (~15-25 min)"
    run ping_scaling_test.py "PING scaling: period vs tau_GABA regression (~30-40 min)"
fi

echo "" | tee -a "$LOG"
echo "==============================================================================" | tee -a "$LOG"
echo "DONE. Everything above is in: $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Read these four lines before anything else:" | tee -a "$LOG"
echo "  1. spectral_null   -> false positives should be 0-3/40, power 20/20." | tee -a "$LOG"
echo "  2. artifact_demo   -> E3: raw pipeline must MISS the real rhythm; notched must find it." | tee -a "$LOG"
echo "  3. ca1_v2          -> theta p < 0.001 in ALL five conditions (positive control)." | tee -a "$LOG"
echo "  4. ping_scaling    -> does the peak move with tau_GABA? That is the paper." | tee -a "$LOG"
echo "==============================================================================" | tee -a "$LOG"
