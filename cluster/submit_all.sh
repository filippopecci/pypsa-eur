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

# Optional throttling: cap simultaneously RUNNING year-jobs to N. Set to 0 to disable.
# Implemented with an LSF job group (bgadd -L), which is the correct mechanism for
# limiting concurrency across independent jobs. (numrun()/numpend() in -w only work
# on job arrays, not name globs.) The SC `macro` application also caps concurrency,
# so this is mostly for predictability / courtesy when sharing project 0588.
MAX_PARALLEL=${MAX_PARALLEL:-0}
JOBGROUP="/fp01525/pypsa-macro"
GROUP_OPT=""
if [[ "$MAX_PARALLEL" -gt 0 ]]; then
    # create the group with the limit, or update the limit if it already exists
    bgadd -L "$MAX_PARALLEL" "$JOBGROUP" 2>/dev/null \
        || bgmod -L "$MAX_PARALLEL" "$JOBGROUP"
    GROUP_OPT="-g $JOBGROUP"
    echo "[info] job group $JOBGROUP limited to $MAX_PARALLEL concurrent jobs"
fi

SUBMITTED=()
for y in "${YEARS[@]}"; do
    JOB_NAME="pypsa-yr${y}"
    # eval lets us combine DEP_OPT + GROUP_OPT cleanly.
    CMD="bsub -J $JOB_NAME -env \"YR=$y\" $DEP_OPT $GROUP_OPT < cluster/run_year.lsf"
    echo "[submit] $CMD"
    eval $CMD
    SUBMITTED+=("$JOB_NAME")
done

echo
echo "Submitted ${#SUBMITTED[@]} jobs."
echo "Monitor with: bjobs -u \$USER -w | grep pypsa-yr"
