#!/bin/bash
#SBATCH --job-name=kvs-chain
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
# Chained RL blocks of one config (rl.sbatch) on Mila's 3-hour short-unkillable slot. Each
# block starts from the chain's latest weights, queues its successor first (afterany), and
# runs save_weights.sh beside the run: every finished weight broadcast becomes the chain's
# latest weights. Weights only: the optimizer state restarts each block. The chain stops
# after BLOCKS blocks or two blocks in a row without new weights; scancel the queued block
# to stop it sooner.
#
#   CHAIN=$SCRATCH/kv-race/rl/<name>/chain REPO=<prime-rl checkout> \
#     CONFIG=configs/alfworld/rl-9b-kvs-reprefill.toml ENVS=alfworld_env sbatch chain.sbatch
#
# Env: CHAIN, REPO, CONFIG (relative to REPO), ENVS (environments/ packages, as in rl.sbatch),
# BLOCKS (default 8), WEIGHTS (a new chain's start: an HF dir with config.json; default the
# Qwen3.5-9B snapshot in the HF cache), VENV (default REPO/.venv, else prime-rl-kvs's),
# SGLANG_VENV, OUT_ROOT (passed to rl.sbatch).
#
# Chain dir: base/ (config + tokenizer files), weights/wNNNN (the latest weights, NNNN =
# trainer updates since the start), latest -> weights/wNNNN, blocks (blocks started),
# fails (blocks in a row without new weights), chain.log.
set -u
C=${CHAIN:?set CHAIN} REPO=${REPO:?set REPO} CONFIG=${CONFIG:?set CONFIG}
BLOCKS=${BLOCKS:-8}
HERE=$REPO/scripts/kv_streams
NAME=$(basename "$CONFIG" .toml)
# A worktree has no venv of its own: use the main checkout's (rl.sbatch puts REPO's src first
# on PYTHONPATH, so the code is still REPO's).
[ -d "$REPO/.venv" ] || export VENV=${VENV:-$SCRATCH/kv-race/prime-rl-kvs/.venv}

if [ ! -e "$C/latest" ]; then
  W=${WEIGHTS:-$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/* | head -1)}
  [ -f "$W/config.json" ] || { echo "no config.json in $W"; exit 1; }
  mkdir -p "$C/weights/w0000" "$C/base"
  cp -rL "$W"/. "$C/weights/w0000/"
  echo 0 > "$C/weights/w0000/.global_step"
  for f in "$W"/*; do
    case "$f" in *.safetensors | *.safetensors.index.json) ;; *) cp -L "$f" "$C/base/" ;; esac
  done
  ln -sfn "$C/weights/w0000" "$C/latest"
  echo 0 > "$C/blocks"; echo 0 > "$C/fails"
  echo "new chain $C from $W" | tee -a "$C/chain.log"
fi

latest=$(readlink -f "$C/latest"); step=$(cat "$latest/.global_step")
if [ -f "$C/last_start_step" ] && [ "$(cat "$C/last_start_step")" = "$step" ]; then
  echo $(( $(cat "$C/fails") + 1 )) > "$C/fails"
else
  echo 0 > "$C/fails"
fi
[ "$(cat "$C/fails")" -ge 2 ] && { echo "chain stopped: two blocks without new weights (at w$step)" | tee -a "$C/chain.log"; exit 0; }
blocks=$(( $(cat "$C/blocks") + 1 ))
[ "$blocks" -gt "$BLOCKS" ] && { echo "chain done: $BLOCKS blocks (at w$step)" | tee -a "$C/chain.log"; exit 0; }
echo "$blocks" > "$C/blocks"; echo "$step" > "$C/last_start_step"
echo "[chain block $blocks/$BLOCKS, job $SLURM_JOB_ID] start w$step $(date +%F' '%T)" | tee -a "$C/chain.log"

# Earlier blocks' broadcasts are superseded by the saved weights.
rm -rf "${OUT_ROOT:-$SCRATCH/kv-race/rl}/$NAME"/*/run/*/broadcasts
export CHAIN REPO CONFIG ENVS BLOCKS
sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL --job-name="$SLURM_JOB_NAME" \
  --output="$(dirname "$C")/%x-%j.out" "$HERE/chain.sbatch"

export STEP0=$step WANDB_MODE=${WANDB_MODE:-online}
ARGS="--model.name $latest ${ARGS:-}" WATCH="bash $HERE/qv/save_weights.sh" bash "$HERE/rl.sbatch"
