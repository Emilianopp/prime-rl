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
# ARM: vllm | sglang | sglang-sessions | kve (KV-Streams turn-window eviction,
# r${MAX_TURNS:-8}-${STRIDE:-1}).  CPR: in-flight episodes per inference
# rank (x2 ranks).  The cell ends once EPISODES (default 512) episodes have
# finished, or at the time limit; summary.json is written either way.
#
# Env: CONTROLLER=1 (let prime-rl's concurrency controller trim below the
# target: the target is then the starting cap and the ceiling), EXTRA_ARGS (extra `rl` arguments, e.g. another model), HARNESS (agent harness id: inline-chat = in-process chat, null = the
# chat program relaunched per turn), REPO (prime-rl checkout), VENV (its environment; another checkout's venv
# runs this checkout's code via PYTHONPATH), SGLANG_VENV (the SGLang fork's
# venv), GRID_OUT (results root). Mila: the DCGM exporter holds ZMQ port 5555.
set -u
ARM=$1 CPR=$2 EPISODES=${3:-512} STEPS=${4:-200}
REPO=${REPO:-$SCRATCH/kv-race/prime-rl-kvs}
VENV=${VENV:-$REPO/.venv}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
GRID_OUT=${GRID_OUT:-$SCRATCH/kv-race/tw_grid}
MAX_TURNS=${MAX_TURNS:-8} STRIDE=${STRIDE:-1} HARNESS=${HARNESS:-inline-chat}
CELL=${ARM}_c${CPR}${CELL_SUFFIX:-}
[ "$ARM" = kve ] && CELL=kve_c${CPR}_r${MAX_TURNS}-${STRIDE}${CELL_SUFFIX:-}
[ "$HARNESS" != inline-chat ] && CELL=${CELL}_h${HARNESS}
[ "${CONTROLLER:-0}" = 1 ] && CELL=${CELL}_ctl
OUT=$GRID_OUT/$CELL/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
INFLIGHT=$((2 * CPR))
MIN_INFLIGHT=$INFLIGHT
[ "${CONTROLLER:-0}" = 1 ] && MIN_INFLIGHT=4
BATCH=$EPISODES
if [ "$BATCH" -lt 512 ]; then BATCH=$(( (EPISODES + 3) / 4 * 4 )); fi

export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export UV_PROJECT_ENVIRONMENT=$VENV
if [ "$VENV" != "$REPO/.venv" ]; then
  export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src:$REPO/environments/textworld_env:$REPO/deps/verifiers${PYTHONPATH:+:$PYTHONPATH}
fi
cd "$REPO"
echo "[$CELL] $(git log --oneline -1)  inflight=$INFLIGHT min_inflight=$MIN_INFLIGHT batch=$BATCH steps=$STEPS harness=$HARNESS extra=${EXTRA_ARGS:-}" | tee "$OUT/cell.txt"

case $ARM in
  vllm) BACKEND=(); ENGINE_PORT=8100 ;;  # the engine, behind the router on 8000
  sglang|sglang-sessions|kve)
    # Node-local copy: concurrent cold imports from scratch have failed with ENOENT.
    rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"
    SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python
    export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13
    ENGINE_PORT=8000  # no router
    BACKEND=(--inference.backend sglang --inference.sglang.python "$SGL_PY")
    [ "$ARM" != sglang ] && BACKEND+=(--inference.sglang.sessions true)
    [ "$ARM" = kve ] && BACKEND+=(--inference.sglang.eviction.max-turns "$MAX_TURNS" --inference.sglang.eviction.stride "$STRIDE")
    ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

# The cell's config: the grid base + this cell's episode log.
sed -e "s|^env.max_episode_steps = .*|env.max_episode_steps = $STEPS\nenv.episode_log = \"$OUT/episodes.jsonl\"|" \
  -e "s|^env.player.harness.id = .*|env.player.harness.id = \"$HARNESS\"|" \
  configs/textworld-grid/rl.toml > "$OUT/rl.toml"

uv run rl @ "$OUT/rl.toml" --clean --output-dir "$OUT/run" \
  --orchestrator.batch-size "$BATCH" \
  --orchestrator.concurrency.initial-inflight "$INFLIGHT" \
  --orchestrator.concurrency.min-inflight "$MIN_INFLIGHT" \
  --orchestrator.concurrency.max-inflight "$INFLIGHT" \
  --monitors.wandb.project textworld-grid-kvs --monitors.wandb.name "$CELL" \
  --rollout-transport.type zmq --rollout-transport.port 5655 \
  "${BACKEND[@]}" ${EXTRA_ARGS:-} > "$OUT/rl.log" 2>&1 &
RL_PID=$!

# Engine gauges and GPU utilization every 15 s until the cell is done.
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 15 > "$OUT/gpu.csv" &
SMI_PID=$!
while kill -0 $RL_PID 2>/dev/null; do
  m=$(curl -s -m 5 "http://localhost:$ENGINE_PORT/metrics") && [ -n "$m" ] && \
    printf '# time %s\n%s\n' "$(date +%s)" \
      "$(grep -E '^(vllm:(kv_cache_usage_perc|num_requests_running|num_requests_waiting|num_preemptions_total|prefix_cache_queries_total|prefix_cache_hits_total|prompt_tokens_total|generation_tokens_total|cache_config_info)|kvs:)' <<<"$m")" >> "$OUT/metrics.prom"
  done_n=$(wc -l < "$OUT/episodes.jsonl" 2>/dev/null || echo 0)
  if [ "$done_n" -ge "$EPISODES" ]; then echo "[$CELL] $done_n episodes finished"; break; fi
  sleep 15
done

pkill -P $RL_PID 2>/dev/null; kill $RL_PID $SMI_PID 2>/dev/null
pkill -u "$USER" -f "$VENV/" 2>/dev/null
pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null
sleep 10
pkill -9 -u "$USER" -f "$VENV/" 2>/dev/null
pkill -9 -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null

"$VENV/bin/python" "$REPO/scripts/textworld_grid/summarize.py" "$OUT" --cell "$CELL" --cpr "$CPR" \
  --arm "$ARM" --episodes "$EPISODES" | tee "$OUT/summary.txt"
