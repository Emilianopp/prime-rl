#!/bin/bash
# Long continuous Craftax eval of RL weights on one cluster, as BLOCKS jobs in a row (3 h each, so
# they backfill fast) that keep playing the SAME games: every block resumes each game from
# $CHAIN_DIR/state (world, counters, the model's memory, and the handoff summary it wrote when the
# previous block's TIME_LIMIT hit), so a game is never reset between blocks. Inference only: the
# SGLang fork (KV-Streams r50-1) + `uv run eval`, with the rl-qv-mem prompt (rlmem-eval.toml).
# Run on the login node, from anywhere:
#
#   cluv run fir -- env WEIGHTS=$SCRATCH/kv-race/cluv/weights/w0049 bash eval_chain.sh
#
# Env: WEIGHTS (HF weights dir, required), BLOCKS (5), TIME_LIMIT (game seconds per block, 8640:
# 5 x 8640 s = 12 h of play), N (games = in-flight episodes, 64), SKIP (first seed, 0), NAME,
# CHAIN_DIR, ACCOUNT (rrg-bengioy-ad). Extra arguments go to every sbatch.
set -eu
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/env.sh"
WEIGHTS=${WEIGHTS:?set WEIGHTS to a HF weights dir}
BLOCKS=${BLOCKS:-5} TIME_LIMIT=${TIME_LIMIT:-8640} N=${N:-64} SKIP=${SKIP:-0}
NAME=${NAME:-$(basename "$WEIGHTS")-$((BLOCKS * TIME_LIMIT / 3600))h}
CHAIN_DIR=${CHAIN_DIR:-$KVS_ROOT/craftax_eval/$NAME}
ACCOUNT=${ACCOUNT:-rrg-bengioy-ad}
case "$KVS_CLUSTER" in
  nibi) CPUS=56 MEM=500G ;;
  rorqual) CPUS=64 MEM=0 ;;
  *) CPUS=48 MEM=500G ;;
esac
[ -f "$WEIGHTS/config.json" ] || { echo "no config.json in $WEIGHTS"; exit 1; }
mkdir -p "$CHAIN_DIR/state"
cd "$HERE"  # job.sh sources $SLURM_SUBMIT_DIR/env.sh
prev=""
for block in $(seq 0 $((BLOCKS - 1))); do
  prev=$(env EXTRA_TOML="$HERE/rlmem-eval.toml" CONTINUOUS=1 FEEDBACK=1 MAX_ACTIONS=4 SEGMENTS=106496:8192 \
    TIME_LIMIT="$TIME_LIMIT" N="$N" CONC="$N" SKIP="$SKIP" WEIGHTS="$WEIGHTS" STATE_DIR="$CHAIN_DIR/state" \
    OUT_ROOT="$CHAIN_DIR" GIT_COMMIT="$(git -C "$REPO" rev-parse HEAD)" \
    sbatch --parsable --account="$ACCOUNT" --gpus-per-node=h100:4 --cpus-per-task="$CPUS" --mem="$MEM" \
    --time=3:00:00 --job-name="$NAME-b$block" --output="$CHAIN_DIR/%x-%j.out" \
    ${prev:+--dependency=afterany:$prev} "$@" job.sh bash "$REPO/scripts/craftax/eval.sbatch")
  echo "block $block: job $prev"
done
echo "chain: $CHAIN_DIR (games in $CHAIN_DIR/state; the last block's summary.txt has every game's totals)"
