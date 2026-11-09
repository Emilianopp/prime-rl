#!/bin/bash
# A judge-RL chain (scripts/kv_streams/qv/chain.sbatch) on one DRAC cluster: 3 h blocks, each
# queueing the next (afterany) through job.sh, all playing the SAME games (env.state_dir =
# $CHAIN/worlds) from the chain's latest weights, until the judge budget is spent (a config
# without a cost_cap runs until stopped: scancel the chain's queued block). Run on the login node:
#
#   cluv run fir -- env CONFIG=configs/craftax/rl-qv-grave-dp.toml BUDGET=30 bash rl_chain.sh
#
# Env: CONFIG (required, relative to the repo), NAME (the config's basename; outputs in
# $KVS_ROOT/rl/$NAME/<job>), BUDGET (judge dollars, 30), WEIGHTS (start weights, an HF dir;
# default the base Qwen3.5-4B in $HF_HOME; only used when the chain is new), CHAIN_DIR
# ($KVS_ROOT/rl/$NAME/chain), ACCOUNT (rrg-bengioy-ad). Re-running it on an existing chain
# just queues another block from its latest weights. The judge's key comes from the repo's .env.
set -eu
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/env.sh"
CONFIG=${CONFIG:?set CONFIG, e.g. configs/craftax/rl-qv-grave.toml}
[ -f "$REPO/$CONFIG" ] || { echo "no $REPO/$CONFIG"; exit 1; }
NAME=${NAME:-$(basename "$CONFIG" .toml)}
CHAIN=${CHAIN_DIR:-$KVS_ROOT/rl/$NAME/chain}
ACCOUNT=${ACCOUNT:-rrg-bengioy-ad}
case "$KVS_CLUSTER" in
  nibi) CPUS=56 MEM=1000G ;;
  rorqual) CPUS=64 MEM=0 ;;
  *) CPUS=48 MEM=1000G ;;
esac
if [ ! -e "$CHAIN/latest" ]; then
  W=${WEIGHTS:-$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.5-4B/snapshots/* | head -1)}
  [ -f "$W/config.json" ] || { echo "no config.json in $W"; exit 1; }
  mkdir -p "$CHAIN/weights/w0000" "$CHAIN/base"
  cp -rL "$W"/. "$CHAIN/weights/w0000/"
  echo 0 > "$CHAIN/weights/w0000/.global_step"
  for f in "$W"/*; do  # config + tokenizer files, copied into every saved weights dir
    case "$f" in *.safetensors | *.safetensors.index.json) ;; *) cp -L "$f" "$CHAIN/base/" ;; esac
  done
  ln -sfn "$CHAIN/weights/w0000" "$CHAIN/latest"
  echo "${BUDGET:-30}" > "$CHAIN/budget"
  : > "$CHAIN/ledgers"
  echo 0 > "$CHAIN/fails"
  echo "new chain $CHAIN from $W, budget \$$(cat "$CHAIN/budget")"
fi
mkdir -p "$KVS_ROOT/rl/$NAME"
export CHAIN REPO VENV CONFIG CHAIN_JOB="$HERE/job.sh" GIT_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
export CHAIN_SBATCH="--account=$ACCOUNT --gpus-per-node=h100:4 --cpus-per-task=$CPUS --mem=$MEM --time=3:00:00 --job-name=$NAME --output=$KVS_ROOT/rl/$NAME/%x-%j.out"
cd "$HERE"  # job.sh loads $SLURM_SUBMIT_DIR/env.sh
# shellcheck disable=SC2086
sbatch --parsable --export=ALL $CHAIN_SBATCH job.sh bash "$REPO/scripts/kv_streams/qv/chain.sbatch"
