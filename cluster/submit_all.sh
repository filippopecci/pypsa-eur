#!/bin/bash
# Submit all 31 weather-year jobs. Run this from a login node.
#
# Order of operations for a fresh CASSANDRA setup:
#   1. bsub < cluster/prefetch_archives.lsf     # ~2-4 h, downloads 14 prebuilt cutouts
#   2. bsub < cluster/build_cutouts.lsf         # builds the 17 missing via CDS
#   3. ./cluster/submit_all.sh                  # 31 weather-year jobs
#
# Or chain them automatically (uncomment in step 3):
#   PREFETCH_ID=$(bsub < cluster/prefetch_archives.lsf | awk '{print $2}' | tr -d '<>')
#   BUILD_ID=$(bsub -w "done($PREFETCH_ID)" < cluster/build_cutouts.lsf | awk '{print $2}' | tr -d '<>')
#   ./cluster/submit_all.sh "$BUILD_ID"

set -euo pipefail

# Optional: wait for the cutout-build job to finish before any year-job runs.
# Pass the cutout-build LSF job ID as argv[1].
DEP_OPT=""
if [[ $# -ge 1 ]]; then
    DEP_OPT="-w done($1)"
    echo "[info] all year-jobs will wait for done($1)"
fi

# Optional throttling: cap simultaneous running year-jobs to N. Set to 0 to disable.
# CASSANDRA's p_macro SC will have its own slot limit; this just stops you
# from queuing more than your fair share at once.
MAX_PARALLEL=${MAX_PARALLEL:-0}

SUBMITTED=()
for y in $(seq 1995 2025); do
    JOB_NAME="pypsa-yr${y}"
    THROTTLE=""
    if [[ "$MAX_PARALLEL" -gt 0 ]]; then
        THROTTLE='-w "numrun(pypsa-yr*) < '"$MAX_PARALLEL"'"'
    fi
    # Build the bsub line. eval lets us combine DEP_OPT + THROTTLE cleanly.
    CMD="bsub -J $JOB_NAME -env \"YR=$y\" $DEP_OPT $THROTTLE < cluster/run_year.lsf"
    echo "[submit] $CMD"
    eval $CMD
    SUBMITTED+=("$JOB_NAME")
done

echo
echo "Submitted ${#SUBMITTED[@]} jobs."
echo "Monitor with: bjobs -u \$USER -w | grep pypsa-yr"
