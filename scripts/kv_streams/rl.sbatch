#!/bin/bash
#SBATCH --job-name=kvs-rl
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
# An RL run (or smoke) of one config on one 4-GPU node, then per-step trainer and
# orchestrator timings: what the trainer and the inference engines each spend.
#
#   CONFIG=configs/textworld/rl-expert-9b.toml STEPS=4 sbatch rl.sbatch
#
# Env: CONFIG (rl toml in REPO), STEPS (max_steps override; default the config's),
# ENVS (environments/ packages on PYTHONPATH, default textworld_env), REPO
# (prime-rl checkout, deps/ submodules checked out), VENV, SGLANG_VENV, OUT_ROOT.
set -u
CONFIG=${CONFIG:?set CONFIG} STEPS=${STEPS:-} ENVS=${ENVS:-textworld_env}
REPO=${REPO:-$SCRATCH/kv-race/prime-rl-kvs}
VENV=${VENV:-$REPO/.venv}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
OUT=${OUT_ROOT:-$SCRATCH/kv-race/rl}/$(basename "$CONFIG" .toml)/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"

export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=${WANDB_MODE:-offline} TOKENIZERS_PARALLELISM=false
export UV_PROJECT_ENVIRONMENT=$VENV
export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src$(for e in $ENVS; do printf ":%s" "$REPO/environments/$e"; done):$REPO/deps/verifiers:$REPO/deps/renderers${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
echo "[rl $CONFIG steps=${STEPS:-config}] $(git log --oneline -1)" | tee "$OUT/cell.txt"

rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"
SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python
export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13

uv run rl @ "$CONFIG" --clean --output-dir "$OUT/run" --inference.sglang.python "$SGL_PY" \
  ${STEPS:+--max-steps "$STEPS"} --rollout-transport.type zmq --rollout-transport.port 5655 > "$OUT/rl.log" 2>&1
echo "rl exit $?"

L=$(ls -d "$OUT"/run/*/logs/attempt_1 2>/dev/null | head -1)
strip() { sed 's/\x1b\[[0-9;]*m//g'; }
echo "== trainer steps"; grep -a -E "Step [0-9]+ \|" "$L/trainer.log" | strip | cut -c1-300 | tee "$OUT/trainer_steps.txt"
echo "== orchestrator steps"; grep -a -E "Step [0-9]+ \|" "$L/orchestrator.log" | strip | cut -c1-300 | tee "$OUT/orchestrator_steps.txt"
pkill -u "$USER" -f "$VENV/" 2>/dev/null; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null
exit 0  # the outcome is in the step logs; pkill's no-match status must not fail the job
