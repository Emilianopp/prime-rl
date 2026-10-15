#!/bin/bash
#SBATCH --job-name=tw-grid
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
# One TextWorld grid cell (configs/textworld-grid/rl.toml) on one 4-GPU node.
#
#   sbatch cell.sbatch ARM CPR [EPISODES] [MAX_EPISODE_STEPS]
#
# ARM: vllm | sglang | sglang-sessions.  CPR: in-flight episodes per inference
# rank (x2 ranks).  The cell ends once EPISODES (default 512) episodes have
# finished, or at the time limit; summary.json is written either way.
#
# Env: REPO (prime-rl checkout), SGLANG_VENV (the SGLang fork's venv),
# GRID_OUT (results root). Mila: the DCGM exporter holds ZMQ port 5555.
set -u
ARM=$1 CPR=$2 EPISODES=${3:-512} STEPS=${4:-200}
REPO=${REPO:-$SCRATCH/kv-race/prime-rl-kvs}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
GRID_OUT=${GRID_OUT:-$SCRATCH/kv-race/tw_grid}
CELL=${ARM}_c${CPR}${CELL_SUFFIX:-}
OUT=$GRID_OUT/$CELL/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
INFLIGHT=$((2 * CPR))
BATCH=$EPISODES
if [ "$BATCH" -lt 512 ]; then BATCH=$(( (EPISODES + 3) / 4 * 4 )); fi

export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
cd "$REPO"
echo "[$CELL] $(git log --oneline -1)  inflight=$INFLIGHT batch=$BATCH steps=$STEPS" | tee "$OUT/cell.txt"

case $ARM in
  vllm) BACKEND=(); ENGINE_PORT=8100 ;;  # the engine, behind the router on 8000
  sglang|sglang-sessions)
    # Node-local copy: concurrent cold imports from scratch have failed with ENOENT.
    rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"
    SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python
    export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13
    ENGINE_PORT=8000  # no router
    BACKEND=(--inference.backend sglang --inference.sglang.python "$SGL_PY")
    [ "$ARM" = sglang-sessions ] && BACKEND+=(--inference.sglang.sessions true)
    ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

# The cell's config: the grid base + this cell's episode log.
sed "s|^env.max_episode_steps = .*|env.max_episode_steps = $STEPS\nenv.episode_log = \"$OUT/episodes.jsonl\"|" \
  configs/textworld-grid/rl.toml > "$OUT/rl.toml"

uv run rl @ "$OUT/rl.toml" --clean --output-dir "$OUT/run" \
  --orchestrator.batch-size "$BATCH" \
  --orchestrator.concurrency.initial-inflight "$INFLIGHT" \
  --orchestrator.concurrency.min-inflight "$INFLIGHT" \
  --orchestrator.concurrency.max-inflight "$INFLIGHT" \
  --monitors.wandb.project textworld-grid-kvs --monitors.wandb.name "$CELL" \
  --rollout-transport.type zmq --rollout-transport.port 5655 \
  "${BACKEND[@]}" > "$OUT/rl.log" 2>&1 &
RL_PID=$!

# Engine gauges and GPU utilization every 15 s until the cell is done.
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 15 > "$OUT/gpu.csv" &
SMI_PID=$!
while kill -0 $RL_PID 2>/dev/null; do
  m=$(curl -s -m 5 "http://localhost:$ENGINE_PORT/metrics") && [ -n "$m" ] && \
    printf '# time %s\n%s\n' "$(date +%s)" \
      "$(grep -E '^vllm:(kv_cache_usage_perc|num_requests_running|num_requests_waiting|num_preemptions_total|prefix_cache_queries_total|prefix_cache_hits_total|prompt_tokens_total|generation_tokens_total|cache_config_info)' <<<"$m")" >> "$OUT/metrics.prom"
  done_n=$(wc -l < "$OUT/episodes.jsonl" 2>/dev/null || echo 0)
  if [ "$done_n" -ge "$EPISODES" ]; then echo "[$CELL] $done_n episodes finished"; break; fi
  sleep 15
done

pkill -P $RL_PID 2>/dev/null; kill $RL_PID $SMI_PID 2>/dev/null
pkill -u "$USER" -f "$REPO/.venv/" 2>/dev/null
pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null
sleep 10
pkill -9 -u "$USER" -f "$REPO/.venv/" 2>/dev/null
pkill -9 -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null

"$REPO/.venv/bin/python" "$REPO/scripts/textworld_grid/summarize.py" "$OUT" --cell "$CELL" --cpr "$CPR" \
  --arm "$ARM" --episodes "$EPISODES" | tee "$OUT/summary.txt"
