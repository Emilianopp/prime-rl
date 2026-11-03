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
# (prime-rl checkout, deps/ submodules checked out), VENV, SGLANG_VENV, OUT_ROOT,
# RUN (a fixed run name for a chain of jobs: one output dir, a checkpoint every CKPT_INTERVAL
# steps (default 1), and each job resumes from the latest complete checkpoint, trainer and
# orchestrator both; a job that finds none starts the run over), CKPT_INTERVAL, CKPT_KEEP (checkpoints kept, default 2).
set -u
CONFIG=${CONFIG:?set CONFIG} STEPS=${STEPS:-} ENVS=${ENVS:-textworld_env} RUN=${RUN:-}
REPO=${REPO:-$SCRATCH/kv-race/prime-rl-kvs}
VENV=${VENV:-$REPO/.venv}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
OUT=${OUT_ROOT:-$SCRATCH/kv-race/rl}/$(basename "$CONFIG" .toml)/${RUN:-${SLURM_JOB_ID:-local}}
mkdir -p "$OUT"

echo "[rl $CONFIG steps=${STEPS:-config}] $(git -C "$REPO" log --oneline -1)" | tee "$OUT/cell.txt"

# Run from node-local copies: one env worker imports for ~50 s off network scratch
# and the elastic pool spawns ~30 at once (minutes before the first episode).
rsync -a --exclude .git --exclude .venv --exclude outputs "$REPO/" "$SLURM_TMPDIR/repo/"
rsync -a "$VENV/" "$SLURM_TMPDIR/prl-venv/"
find "$SLURM_TMPDIR/prl-venv/bin" -maxdepth 1 -type f -exec sed -i "1s|^#!$VENV/|#!$SLURM_TMPDIR/prl-venv/|" {} +
REPO=$SLURM_TMPDIR/repo VENV=$SLURM_TMPDIR/prl-venv

export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=${WANDB_MODE:-offline} TOKENIZERS_PARALLELISM=false
export UV_PROJECT_ENVIRONMENT=$VENV
export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src$(for e in $ENVS; do printf ":%s" "$REPO/environments/$e"; done):$REPO/deps/verifiers:$REPO/deps/renderers${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"

rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"
SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python
export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13

START=(--clean)
if [ -n "$RUN" ]; then
  START=(--run.name "$RUN" --ckpt.interval "${CKPT_INTERVAL:-1}" --ckpt.keep-last "${CKPT_KEEP:-2}")
  # Resume from the newest step that holds BOTH halves: a job cut off by its wall clock can leave
  # the orchestrator's next checkpoint without the trainer's, and a bare --resume picks that
  # step (FileNotFoundError: .../step_N/trainer) on every later job of the chain.
  STEP=$(ls -d "$OUT/run/$RUN/checkpoints"/step_* 2>/dev/null | sort -V | while read -r d; do
    [ -d "$d/trainer" ] && [ -d "$d/orchestrator" ] && basename "$d"; done | tail -1)
  if [ -n "$STEP" ]; then START+=(--resume.step "${STEP#step_}"); else START+=(--clean); fi
  echo "run $RUN: ${START[*]} (job ${SLURM_JOB_ID:-local})" | tee -a "$OUT/cell.txt"
fi
uv run rl @ "$CONFIG" "${START[@]}" --output-dir "$OUT/run" --inference.sglang.python "$SGL_PY" \
  ${STEPS:+--max-steps "$STEPS"} --rollout-transport.type zmq --rollout-transport.port 5655 > "$OUT/rl.log" 2>&1
echo "rl exit $?"

L=$(ls -d "$OUT"/run/*/logs/attempt_* 2>/dev/null | sort -V | tail -1)
strip() { sed 's/\x1b\[[0-9;]*m//g'; }
echo "== trainer steps"; grep -a -E "Step [0-9]+ \|" "$L/trainer.log" | strip | cut -c1-300 | tee "$OUT/trainer_steps.txt"
echo "== orchestrator steps"; grep -a -E "Step [0-9]+ \|" "$L/orchestrator.log" | strip | cut -c1-300 | tee "$OUT/orchestrator_steps.txt"
pkill -u "$USER" -f "$VENV/" 2>/dev/null; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null
exit 0  # the outcome is in the step logs; pkill's no-match status must not fail the job
