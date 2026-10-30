#!/bin/bash
# Continuous Craftax on the same N worlds for BLOCKS jobs in a row: each block is
# eval.sbatch with CONTINUOUS=1 and STATE_DIR, so every game resumes where the last
# block stopped (game state, counters, and the handoff summary the model wrote at the
# block's TIME_LIMIT). A block starts once the previous one ends, whatever its outcome
# (a failed block's games resume from their last periodic save). Every other eval.sbatch
# knob (REPO, VENV, FEEDBACK, MAX_ACTIONS, KEEP, ...) passes through the environment;
# arguments go to sbatch.
#
#   BLOCKS=5 TIME_LIMIT=8640 N=64 REPO=<prime-rl checkout> scripts/craftax/continuous_chain.sh --exclude=<nodes>
#
# Runs land in $CHAIN_DIR/<cell>/<job id>, the saved games in $CHAIN_DIR/state; the last
# block's episodes.jsonl holds every game's totals.
set -eu
BLOCKS=${BLOCKS:-5} TIME_LIMIT=${TIME_LIMIT:-8640} N=${N:-64}
CHAIN_DIR=${CHAIN_DIR:-${OUT_ROOT:-$SCRATCH/kv-race/craftax_eval}/chain-$(date +%Y%m%d-%H%M)}
SCRIPT=$(dirname "$(readlink -f "$0")")/eval.sbatch
mkdir -p "$CHAIN_DIR/state"
prev=""
for block in $(seq 0 $((BLOCKS - 1))); do
  prev=$(CONTINUOUS=1 N=$N TIME_LIMIT=$TIME_LIMIT STATE_DIR=$CHAIN_DIR/state OUT_ROOT=$CHAIN_DIR \
    sbatch --parsable --job-name="craftax-chain-$block" ${prev:+--dependency=afterany:$prev} "$@" "$SCRIPT")
  echo "block $block: job $prev"
done
echo "chain: $CHAIN_DIR"
