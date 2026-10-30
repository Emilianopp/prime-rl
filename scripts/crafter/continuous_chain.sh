#!/bin/bash
# Continuous Crafter, inference only, as a chain of 3-hour short-unkillable blocks:
# each block is kve_eval.sbatch with CONTINUOUS=1 (configs/crafter/eval-continuous.toml)
# on its own N world seeds, and starts once the previous one ends. An episode lives
# inside its block (TIME_LIMIT, default the config's 2.5 h), so BLOCKS blocks play
# BLOCKS x N episodes. Every other kve_eval.sbatch knob (REPO, VENV, KEEP, REASONING,
# THINKING, ...) passes through the environment; arguments go to sbatch.
#
#   BLOCKS=4 N=64 REPO=<prime-rl checkout> scripts/crafter/continuous_chain.sh --exclude=<nodes>
set -eu
BLOCKS=${BLOCKS:-4} N=${N:-64} CONC=${CONC:-$N} SKIP0=${SKIP:-0}
SCRIPT=$(dirname "$(readlink -f "$0")")/kve_eval.sbatch
prev=""
for block in $(seq 0 $((BLOCKS - 1))); do
  skip=$((SKIP0 + block * N))
  prev=$(CONTINUOUS=1 N=$N CONC=$CONC SKIP=$skip sbatch --parsable --job-name="crafter-continuous-$block" \
    ${prev:+--dependency=afterany:$prev} "$@" "$SCRIPT")
  echo "block $block: job $prev, seeds $skip..$((skip + N - 1))"
done
