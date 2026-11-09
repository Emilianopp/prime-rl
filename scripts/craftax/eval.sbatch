#!/bin/bash
#SBATCH --job-name=craftax-eval
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
# Inference-only full Craftax (craftax-env, GlyphBench's glyph observations): SGLang
# (4 data-parallel engines, KV-Streams) + `uv run eval` with the renderer client, then
# the episode summary.
#
#   sbatch scripts/craftax/eval.sbatch
#
# Env: KEEP (eviction max_turns, default 50; "full": eviction off, plain SGLang), N
# (episodes, default 64), CONC (in-flight episodes, default 64), SKIP (skip the first
# SKIP tasks: disjoint seeds across jobs), TURN_TOKENS (max tokens per reply, default
# 2048; also the budget the system prompt states), NO_THINKING (1: the renderer's
# non-thinking mode and GlyphBench's action-only format), TIME_LIMIT (seconds per
# episode, default the config's 9000), STEPS (turns per episode, default 10000),
# SEGMENTS (B:BS, default the config's 131072:32768; "off": one trace per episode),
# CONTINUOUS (1: configs/craftax/eval-continuous.toml - respawn on death, no turn limit,
# achievement reward; STEPS then sets a turn limit again), FEEDBACK (1: env.action_feedback - the
# message says what each action did), MAX_ACTIONS (N > 1: up to N actions per reply), REPO
# (prime-rl checkout, deps/ submodules checked out), VENV, SGLANG_VENV, CRAFTAX_PY (a
# --target dir with jax[cpu] and the Craftax fork, for a venv without them), OUT_ROOT,
# STATE_DIR (env.state_dir: resume each game from there and save it back - a chain of
# jobs on the same worlds, see continuous_chain.sh), MEMORY (N: env.player.memory with
# entries capped at N tokens - the model's own memory, kept across segments and resumes).
set -u
KEEP=${KEEP:-50} N=${N:-64} CONC=${CONC:-64} SKIP=${SKIP:-0} TURN_TOKENS=${TURN_TOKENS:-2048} NO_THINKING=${NO_THINKING:-}
TIME_LIMIT=${TIME_LIMIT:-} STEPS=${STEPS:-} SEGMENTS=${SEGMENTS:-} CONTINUOUS=${CONTINUOUS:-} FEEDBACK=${FEEDBACK:-} MAX_ACTIONS=${MAX_ACTIONS:-} CRAFTAX_PY=${CRAFTAX_PY:-} STATE_DIR=${STATE_DIR:-} MEMORY=${MEMORY:-}
REPO=${REPO:-$SCRATCH/kv-race/prime-rl-kvs}
VENV=${VENV:-$REPO/.venv}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
OUT=${OUT_ROOT:-$SCRATCH/kv-race/craftax_eval}/r${KEEP}-1_n${N}_c${CONC}_tt${TURN_TOKENS}${NO_THINKING:+_nothink}${TIME_LIMIT:+_tl$TIME_LIMIT}${STEPS:+_steps$STEPS}${SEGMENTS:+_seg-${SEGMENTS//:/-}}${CONTINUOUS:+_continuous}${FEEDBACK:+_feedback}${MAX_ACTIONS:+_act$MAX_ACTIONS}${MEMORY:+_mem$MEMORY}_skip$SKIP/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"

export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false VLLM_API_KEY=EMPTY JAX_PLATFORMS=cpu
export UV_PROJECT_ENVIRONMENT=$VENV
export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src:$REPO/environments/craftax_env:$REPO/deps/verifiers:$REPO/deps/renderers${CRAFTAX_PY:+:$CRAFTAX_PY}${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
# API keys for plugged judges (e.g. GEMINI_API_KEY for craftax_env.judge); .env is gitignored.
[ -f "$REPO/.env" ] && { set -a; . "$REPO/.env"; set +a; }
echo "[craftax eval r$KEEP-1 n$N c$CONC tt$TURN_TOKENS thinking=${NO_THINKING:+off}${NO_THINKING:-on} time_limit=${TIME_LIMIT:-config} steps=${STEPS:-config} segments=${SEGMENTS:-config} continuous=${CONTINUOUS:-off} feedback=${FEEDBACK:-off} max_actions=${MAX_ACTIONS:-1} state=${STATE_DIR:-none} memory=${MEMORY:-off}] skip=$SKIP $(git log --oneline -1)" | tee "$OUT/cell.txt"

rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"
SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python
export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13

if [ "$KEEP" = full ]; then
  sed '/^\[sglang.eviction\]/,$d' configs/crafter/inference-kve.toml > "$OUT/inference.toml"
  EVICT=()
else
  cp configs/crafter/inference-kve.toml "$OUT/inference.toml"
  EVICT=(--sglang.eviction.max-turns "$KEEP")
fi
uv run inference @ "$OUT/inference.toml" --sglang.python "$SGL_PY" "${EVICT[@]}" > "$OUT/inference.log" 2>&1 &
INFER=$!
for _ in $(seq 120); do
  curl -sf -m 5 http://localhost:8000/health > /dev/null && break
  kill -0 $INFER 2>/dev/null || { echo "inference died"; tail -30 "$OUT/inference.log"; exit 1; }
  sleep 10
done
curl -sf -m 5 http://localhost:8000/health > /dev/null || { echo "inference not up after 20 min"; tail -30 "$OUT/inference.log"; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/"; exit 1; }
echo "inference up $(date +%T)"

( while sleep 30; do curl -s -m 5 http://localhost:8000/metrics | grep "^kvs:" > "$OUT/kvs.prom.tmp" && mv "$OUT/kvs.prom.tmp" "$OUT/kvs.prom"; done ) &
POLL=$!
EVAL_TOML=configs/craftax/eval.toml
[ -n "$CONTINUOUS" ] && EVAL_TOML=configs/craftax/eval-continuous.toml
sed "s|^env.max_episode_steps = .*|&\nenv.episode_log = \"$OUT/episodes.jsonl\"|" "$EVAL_TOML" > "$OUT/eval.toml"
sed -i "s/^max_completion_tokens = .*/max_completion_tokens = $TURN_TOKENS/; s/^env.taskset.max_output_tokens = [0-9]*/env.taskset.max_output_tokens = $TURN_TOKENS/" "$OUT/eval.toml"
[ -n "$FEEDBACK" ] && sed -i "s|^env.max_episode_steps = .*|&\nenv.action_feedback = true|" "$OUT/eval.toml"
[ -n "$STATE_DIR" ] && mkdir -p "$STATE_DIR" && sed -i "s|^env.max_episode_steps = .*|&\nenv.state_dir = \"$STATE_DIR\"|" "$OUT/eval.toml"
[ -n "$MEMORY" ] && sed -i "s|^env.player.harness.id = .*|&\nenv.player.memory.max_key_tokens = $MEMORY|" "$OUT/eval.toml"
[ -n "$MAX_ACTIONS" ] && sed -i "s|^env.taskset.split = .*|&\nenv.taskset.max_actions_per_turn = $MAX_ACTIONS|" "$OUT/eval.toml"
[ -n "$NO_THINKING" ] && sed -i "s/^enable_thinking = true/enable_thinking = false/; s|^env.taskset.split = .*|&\nenv.taskset.enable_reasoning = false|" "$OUT/eval.toml"
[ -n "$TIME_LIMIT" ] && sed -i "s/^env.max_episode_seconds = .*/env.max_episode_seconds = $TIME_LIMIT/" "$OUT/eval.toml"
[ -n "$STEPS" ] && sed -i "s/^env.max_episode_steps = .*/env.max_episode_steps = $STEPS/" "$OUT/eval.toml"
if [ "$SEGMENTS" = off ]; then
  sed -i "/^env.player.segments/d; s|^env.player.harness.id = .*|env.player.max_total_tokens = 256000\n&|" "$OUT/eval.toml"
elif [ -n "$SEGMENTS" ]; then
  IFS=: read -r SEG_B SEG_S <<< "$SEGMENTS"
  sed -i "s/^env.player.segments.new_tokens = .*/env.player.segments.new_tokens = $SEG_B/; s/^env.player.segments.summary_tokens = .*/env.player.segments.summary_tokens = $SEG_S/" "$OUT/eval.toml"
fi
uv run eval @ "$OUT/eval.toml" --clean --output-dir "$OUT/run" --dashboard false \
  --select.skip "$SKIP" --select.limit "$N" --concurrency.min-inflight "$CONC" --concurrency.max-inflight "$CONC" \
  --concurrency.initial-inflight "$CONC" > "$OUT/eval.log" 2>&1
echo "eval exit $? $(date +%T)"

kill $POLL 2>/dev/null; echo "== kvs counters"; cat "$OUT/kvs.prom" 2>/dev/null
echo "== episodes"; $VENV/bin/python scripts/craftax/summarize_episodes.py "$OUT/episodes.jsonl" | tee "$OUT/summary.txt"
kill $INFER 2>/dev/null; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null
exit 0  # the outcome is in summary.txt; pkill's no-match status must not fail the job
