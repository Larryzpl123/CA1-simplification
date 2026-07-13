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

set -u
cd "$(dirname "$0")" || exit 1

PY="${PYTHON:-python3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="results_${STAMP}.log"

# ---- expected fingerprints (sha256, first 12 hex chars) ----------------------
declare -a FILES=(spectral_null.py ca1_v2.py artifact_demo.py ping_scaling_test.py)
declare -a WANT=(c8f356eaf686     b18a329f5027 a6d20715722c     346c8a9d255b)

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
    echo "   [${1} finished in $(( $(date +%s) - t0 ))s]" | tee -a "$LOG"
}

# ONLY=ping bash run_all.sh   -> re-run just the PING test (the other three are
# unchanged since the last run and their outputs are already logged).
if [ "${ONLY:-all}" = "ping" ]; then
    run ping_scaling_test.py "PING scaling: period vs tau_GABA regression (~30-40 min)"
else
    run spectral_null.py     "validate the null test itself: false-positive rate and power"
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
echo "  2. artifact_demo   -> E3 ground-truth 40 Hz must be found AT 40.0 Hz." | tee -a "$LOG"
echo "  3. ca1_v2          -> theta p < 0.001 in ALL five conditions (positive control)." | tee -a "$LOG"
echo "  4. ping_scaling    -> does the peak move with tau_GABA? That is the paper." | tee -a "$LOG"
echo "==============================================================================" | tee -a "$LOG"
