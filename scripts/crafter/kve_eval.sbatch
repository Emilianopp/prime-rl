#!/bin/bash
#SBATCH --job-name=crafter-eval
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
# Inference-only Crafter baseline: SGLang (4 data-parallel engines, KV-Streams) +
# `uv run eval` with the renderer client, then the episode summary.
#
#   sbatch kve_eval.sbatch
#
# Env: KEEP (eviction max_turns, default 50), N (episodes, default 256), CONC
# (in-flight episodes, default 128), REPO (prime-rl checkout; deps/ submodules
# checked out), VENV, SGLANG_VENV, OUT_ROOT.
set -u
KEEP=${KEEP:-50} N=${N:-256} CONC=${CONC:-128}
REPO=${REPO:-$SCRATCH/kv-race/prime-rl-kvs}
VENV=${VENV:-$REPO/.venv}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
OUT=${OUT_ROOT:-$SCRATCH/kv-race/crafter_eval}/r${KEEP}-1_n${N}_c${CONC}/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"

export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false VLLM_API_KEY=EMPTY
export UV_PROJECT_ENVIRONMENT=$VENV
export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src:$REPO/environments/crafter_env:$REPO/deps/verifiers:$REPO/deps/renderers${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
echo "[crafter eval r$KEEP-1 n$N c$CONC] $(git log --oneline -1)" | tee "$OUT/cell.txt"

rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"
SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python
export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13

uv run inference @ configs/crafter/inference-kve.toml --sglang.python "$SGL_PY" \
  --sglang.eviction.max-turns "$KEEP" > "$OUT/inference.log" 2>&1 &
INFER=$!
until curl -sf -m 5 http://localhost:8000/health > /dev/null; do
  kill -0 $INFER 2>/dev/null || { echo "inference died"; tail -30 "$OUT/inference.log"; exit 1; }
  sleep 10
done
echo "inference up $(date +%T)"

( while sleep 30; do curl -s -m 5 http://localhost:8000/metrics | grep "^kvs:" > "$OUT/kvs.prom.tmp" && mv "$OUT/kvs.prom.tmp" "$OUT/kvs.prom"; done ) &
POLL=$!
sed "s|^env.max_episode_steps = .*|&\nenv.episode_log = \"$OUT/episodes.jsonl\"|" configs/crafter/eval-kve.toml > "$OUT/eval.toml"
uv run eval @ "$OUT/eval.toml" --clean --output-dir "$OUT/run" --dashboard false \
  --select.limit "$N" --concurrency.min-inflight "$CONC" --concurrency.max-inflight "$CONC" > "$OUT/eval.log" 2>&1
echo "eval exit $? $(date +%T)"

kill $POLL 2>/dev/null; echo "== kvs counters"; cat "$OUT/kvs.prom" 2>/dev/null
echo "== episodes"; $VENV/bin/python scripts/crafter/summarize_episodes.py "$OUT/episodes.jsonl" | tee "$OUT/summary.txt"
kill $INFER 2>/dev/null; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null
exit 0  # the outcome is in summary.txt; pkill's no-match status must not fail the job
