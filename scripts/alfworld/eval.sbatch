#!/bin/bash
#SBATCH --job-name=alfworld-eval
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
# ALFWorld inference-only eval: SGLang (4 data-parallel engines, KV-Streams sessions)
# + `uv run eval` with the renderer client, then per-task-type success.
#
#   KEEP=full sbatch eval.sbatch
#
# Env: KEEP (turn window, default 10; "full": no eviction), SPLIT (valid_seen,
# valid_unseen or train), N (episodes, default all of the split), CONC (in-flight
# episodes, default 64), DATA (ALFWorld data dir), REPO (prime-rl checkout, deps/
# submodules checked out), VENV, SGLANG_VENV, OUT_ROOT.
set -u
KEEP=${KEEP:-10} SPLIT=${SPLIT:-valid_seen} N=${N:-100000} CONC=${CONC:-64}
DATA=${DATA:-$SCRATCH/kv-race/alfworld_data}
REPO=${REPO:-$SCRATCH/kv-race/prime-rl-kvs}
VENV=${VENV:-$REPO/.venv}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
OUT=${OUT_ROOT:-$SCRATCH/kv-race/alfworld_eval}/${SPLIT}_r${KEEP}_c${CONC}/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"

export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false VLLM_API_KEY=EMPTY
export UV_PROJECT_ENVIRONMENT=$VENV
export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src:$REPO/environments/alfworld_env:$REPO/deps/verifiers:$REPO/deps/renderers${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
echo "[alfworld eval $SPLIT r$KEEP c$CONC] $(git log --oneline -1)" | tee "$OUT/cell.txt"

rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"
SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python
export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13

if [ "$KEEP" = full ]; then
  sed '/^\[sglang.eviction\]/,$d' configs/alfworld/inference.toml > "$OUT/inference.toml"
else
  sed "s/^max_turns = .*/max_turns = $KEEP/" configs/alfworld/inference.toml > "$OUT/inference.toml"
fi
uv run inference @ "$OUT/inference.toml" --sglang.python "$SGL_PY" > "$OUT/inference.log" 2>&1 &
INFER=$!
for _ in $(seq 120); do  # the front end can outlive crashed engines: give up after 20 minutes
  curl -sf -m 5 http://localhost:8000/health > /dev/null && break
  kill -0 $INFER 2>/dev/null || { echo "inference died"; tail -30 "$OUT/inference.log"; exit 1; }
  sleep 10
done
curl -sf -m 5 http://localhost:8000/health > /dev/null || { echo "inference not up after 20 min"; tail -30 "$OUT/inference.log"; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/"; exit 1; }
echo "inference up $(date +%T)"

sed -e "s|^env.taskset.dataset_path = .*|env.taskset.dataset_path = \"$DATA\"|" \
    -e "s|^env.taskset.split = .*|env.taskset.split = \"$SPLIT\"\nenv.episode_log = \"$OUT/episodes.jsonl\"|" \
    configs/alfworld/eval.toml > "$OUT/eval.toml"
uv run eval @ "$OUT/eval.toml" --clean --output-dir "$OUT/run" --dashboard false --select.limit "$N" \
  --concurrency.min-inflight "$CONC" --concurrency.max-inflight "$CONC" --concurrency.initial-inflight "$CONC" > "$OUT/eval.log" 2>&1
echo "eval exit $? $(date +%T)"

$VENV/bin/python scripts/alfworld/summarize.py "$OUT/episodes.jsonl" | tee "$OUT/summary.txt"
kill $INFER 2>/dev/null; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null
exit 0  # the outcome is in summary.txt; pkill's no-match status must not fail the job
