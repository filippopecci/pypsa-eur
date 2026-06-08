#!/bin/bash
# Submit one network-build job per weather year. Run this from a login node.
#
# Archive mode: cutouts are the pre-built sarah3-era5 files retrieved from
# data.pypsa.org (or already staged in data/cutout/archive/v1.0/). No cutout
# build step is needed, so just run the year jobs directly:
#   ./cluster/submit_all.sh
#
# Optionally throttle concurrency:
#   MAX_PARALLEL=4 ./cluster/submit_all.sh

set -euo pipefail

# The 14 weather years pypsa-eur publishes as pre-built sarah3-era5 cutouts.
YEARS=(1995 1996 2008 2009 2010 2012 2013 2019 2020 2021 2022 2023 2024 2025)

# Optional: make all year-jobs wait for a prior LSF job. Pass its ID as argv[1].
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
for y in "${YEARS[@]}"; do
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
