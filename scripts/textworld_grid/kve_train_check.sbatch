#!/bin/bash
#SBATCH --job-name=kve-train
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
# Train-inference mismatch check for KV-Streams training on TextWorld with
# Qwen3.5-4B: a few real RL steps (2 SGLang ranks + 2 trainer GPUs), then the
# trainer's per-step mismatch KL, split by evicted context.
#
#   sbatch kve_train_check.sbatch ARM [STEPS] [CPR]
#
# ARM: kve        KV-Streams r8-1 rollouts, trainer replays them with the kv_death mask
#      kve-dense  same rollouts, trainer ignores kv_death (negative control)
#      full       full-context rollouts (plain SGLang), the mismatch floor
#
# Env: BATCH (episodes per step, default 128), MAMBA (DeltaNet state slots per
# engine, >= 3 x CPR; default 700), REPO (prime-rl checkout; its deps/ submodules
# must be checked out), VENV, SGLANG_VENV, OUT_ROOT.
#
# Recommended KV-Streams setting (Qwen3.5-4B, 2x A100-80GB inference): the
# concurrency controller on with a c256 ceiling, which it trims to what the KV
# pool holds:  CPR=256 BATCH=512 MAMBA=840.
set -u
ARM=$1 STEPS=${2:-4} CPR=${3:-48} BATCH=${BATCH:-128} MAMBA=${MAMBA:-700}
REPO=${REPO:-$SCRATCH/kv-race/prime-rl-kvs}
VENV=${VENV:-$REPO/.venv}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
OUT=${OUT_ROOT:-$SCRATCH/kv-race/kve_train}/${ARM}_c${CPR}_b${BATCH}/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
INFLIGHT=$((2 * CPR))

export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export UV_PROJECT_ENVIRONMENT=$VENV
# This checkout's code and pinned verifiers/renderers, over the venv's editable installs.
export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src:$REPO/environments/textworld_env:$REPO/deps/verifiers:$REPO/deps/renderers${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
echo "[$ARM c$CPR] $(git log --oneline -1) steps=$STEPS batch=$BATCH mamba=$MAMBA" | tee "$OUT/cell.txt"

rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"
SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python
export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13

MODEL=(--model.name Qwen/Qwen3.5-4B --orchestrator.renderer.name qwen3.5 --orchestrator.renderer.enable-thinking false
       --inference.vllm.gpu-memory-utilization 0.9 --inference.sglang.max-mamba-cache-size "$MAMBA")
BACKEND=(--inference.backend sglang --inference.sglang.python "$SGL_PY")
case $ARM in
  kve|kve-dense)
    BACKEND+=(--inference.sglang.sessions true --inference.sglang.eviction.max-turns 8 --inference.sglang.eviction.stride 1
              --inference.sglang.mamba-radix-cache-strategy no_buffer --inference.sglang.disable-overlap-schedule true)
    [ "$ARM" = kve-dense ] && BACKEND+=(--trainer.model.debug.ignore-kv-death true) ;;
  full) ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

sed "s|^env.max_episode_steps = .*|env.max_episode_steps = 200\nenv.episode_log = \"$OUT/episodes.jsonl\"|" \
  configs/textworld-grid/rl.toml > "$OUT/rl.toml"

# KV-Streams counters (evicting calls, evicted tokens, reopens) while the engine is up.
( while sleep 30; do curl -s -m 5 http://localhost:8000/metrics | grep "^kvs:" > "$OUT/kvs.prom.tmp" && mv "$OUT/kvs.prom.tmp" "$OUT/kvs.prom"; done ) &
POLL=$!
uv run rl @ "$OUT/rl.toml" --clean --output-dir "$OUT/run" \
  --max-steps "$STEPS" --orchestrator.batch-size "$BATCH" \
  --orchestrator.concurrency.initial-inflight "$INFLIGHT" \
  --orchestrator.concurrency.min-inflight 4 \
  --orchestrator.concurrency.max-inflight "$INFLIGHT" \
  --monitors.wandb.project kve-train-check --monitors.wandb.name "${ARM}_c${CPR}" \
  --rollout-transport.type zmq --rollout-transport.port 5655 \
  "${MODEL[@]}" "${BACKEND[@]}" > "$OUT/rl.log" 2>&1
echo "rl exit $?"

L=$(ls -d "$OUT"/run/*/logs/attempt_1 2>/dev/null | head -1)
strip() { sed 's/\x1b\[[0-9;]*m//g'; }
echo "== trainer steps"; grep -E "Step [0-9]+ \|" "$L/trainer.log" | strip | cut -c1-260 | tee "$OUT/steps.txt"
echo "== orchestrator steps"; grep -E "^.*Step [0-9]+ \|" "$L/orchestrator.log" | strip | cut -c1-200
kill $POLL 2>/dev/null; echo "== kvs counters"; cat "$OUT/kvs.prom" 2>/dev/null
pkill -u "$USER" -f "$VENV/" 2>/dev/null; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null
exit 0  # the run's outcome is in steps.txt; pkill's no-match status must not fail the job
