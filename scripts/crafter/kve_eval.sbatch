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
# Env: KEEP (eviction max_turns, default 50; "full": the same session path with
# eviction off, i.e. full context), N (episodes, default 256), CONC
# (in-flight episodes, default 128), RESPAWN (unset: death ends the episode;
# AT:INVENTORY:PENALTY[:safe], e.g. spawn:drop_all:0.05, respawns, see the env's
# on_death / respawn_at / death_inventory / death_penalty; :safe sets respawn_safe), NOTEBOOK (1: the
# model gets the notebook tools, memories + scratchpad), WORKSPACE (both, ledger
# or files: the queryable workspace, with the env-written ledger, the model's
# files or both), EXTRAS (1: observation extras - creatures, nearest resources,
# available crafting, damage; EXTRAS=perception: without available crafting), REASONING (1: 2-4 sentences of reasoning before
# each action), RELAXED (1: the memory arms' reply format without their tools),
# SILENT_MALFORMED (1: repeat the last action silently on a reply without an
# action tag, as before the notice), MIN_TOKENS (sample at least this many tokens
# per reply: no end-of-message right after a lone "<"), FP32_STATE (1: keep the
# DeltaNet state in fp32, as training does; default bf16 for capacity), WEIGHTS (an
# HF safetensors dir, e.g. an SFT run's weights/step_N: loaded into the server
# through /update_weights before the eval), SKIP (skip the first SKIP tasks: disjoint
# task ranges across jobs), CONC_MIN (let the concurrency controller trim down to
# this under KV pressure instead of pinning CONC: full-context sessions otherwise
# fill the KV cache and deadlock; trimmed episodes are dropped), MAX_TOKENS (end
# episodes at this many trace tokens instead of 256000), MAMBA_SLOTS (DeltaNet state
# slots per engine, default 400: ~3 per in-flight episode is enough), EP_STEPS (model
# turns per episode instead of Crafter's 10,000), REPO (prime-rl checkout; deps/ submodules
# checked out), VENV, SGLANG_VENV, OUT_ROOT.
# Summary segments + in-env beam (arm A of docs/crafter-kv-streams-summary-segments.md):
# TREE_DEPTH (> 0 switches to configs/crafter/eval-tree.toml: each task is a root whose
# tree is played inside one run), ROOTS (tasks, default 8; N is ignored), BRANCHES
# (per node, default 4), KEEP_BRANCHES (per depth, default 1), SEGMENT_TURNS (default
# 400), SEGMENT_TOKENS (default 96000), SUMMARY_TOKENS (default 32768), JUDGE_MODEL
# (default: the config's), JUDGE_MODE (single | group: one comparative call per node), CUT_ON_DEATH (1: a death ends the segment after the respawn),
# GEMINI_API_KEY (or ~/.gemini_key; unset: the judge falls back to the achievement delta).
# RESPAWN defaults to last_base:drop_materials:0.05:safe there (last_base: the most recent of the
# last sleep and the last placed table).
set -u
TREE_DEPTH=${TREE_DEPTH:-0} ROOTS=${ROOTS:-8} BRANCHES=${BRANCHES:-4} KEEP_BRANCHES=${KEEP_BRANCHES:-1} SEGMENT_TURNS=${SEGMENT_TURNS:-400} SEGMENT_TOKENS=${SEGMENT_TOKENS:-96000} SUMMARY_TOKENS=${SUMMARY_TOKENS:-32768} JUDGE_MODEL=${JUDGE_MODEL:-} CUT_ON_DEATH=${CUT_ON_DEATH:-} JUDGE_MODE=${JUDGE_MODE:-}
SAFE_RADIUS=${SAFE_RADIUS:-} JUDGE_BACKEND=${JUDGE_BACKEND:-}  # tree runs: tiles with no hostile around the respawn cell (config: 2; the RL runs use 8); judge backend openai (Gemini, the config) | capi (Copilot login, JUDGE_MODEL from its list)
if [ "$TREE_DEPTH" -gt 0 ]; then N=$ROOTS CONC=${CONC:-$ROOTS} RESPAWN=${RESPAWN:-last_base:drop_materials:0.05:safe}; fi
[ "$TREE_DEPTH" = 0 ] && TREE_DEPTH=
[ -z "${GEMINI_API_KEY:-}" ] && [ -r "$HOME/.gemini_key" ] && export GEMINI_API_KEY=$(cat "$HOME/.gemini_key")
KEEP=${KEEP:-50} N=${N:-256} CONC=${CONC:-128} RESPAWN=${RESPAWN:-} NOTEBOOK=${NOTEBOOK:-} WORKSPACE=${WORKSPACE:-} EXTRAS=${EXTRAS:-} REASONING=${REASONING:-} RELAXED=${RELAXED:-} SILENT_MALFORMED=${SILENT_MALFORMED:-} MIN_TOKENS=${MIN_TOKENS:-} FP32_STATE=${FP32_STATE:-} WEIGHTS=${WEIGHTS:-} SKIP=${SKIP:-0} CONC_MIN=${CONC_MIN:-} MAX_TOKENS=${MAX_TOKENS:-} MAMBA_SLOTS=${MAMBA_SLOTS:-} EP_STEPS=${EP_STEPS:-}
REPO=${REPO:-$SCRATCH/kv-race/prime-rl-kvs}
VENV=${VENV:-$REPO/.venv}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
OUT=${OUT_ROOT:-$SCRATCH/kv-race/crafter_eval}/${TREE_DEPTH:+tree${TREE_DEPTH}_b${BRANCHES}k${KEEP_BRANCHES}_s${SEGMENT_TURNS}_}r${KEEP}-1_n${N}_c${CONC}${RESPAWN:+_respawn-${RESPAWN//:/-}}${NOTEBOOK:+_notebook}${WORKSPACE:+_workspace-$WORKSPACE}${EXTRAS:+_extras}${EXTRAS:+${EXTRAS#1}}${REASONING:+_reasoning}${RELAXED:+_relaxed}${SILENT_MALFORMED:+_silent}${MIN_TOKENS:+_min$MIN_TOKENS}${FP32_STATE:+_fp32}_skip$SKIP${WEIGHTS:+_w-$(basename "$(dirname "$(dirname "$WEIGHTS")")")}/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"

export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false VLLM_API_KEY=EMPTY
export UV_PROJECT_ENVIRONMENT=$VENV
export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src:$REPO/environments/crafter_env:$REPO/deps/verifiers:$REPO/deps/renderers${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
echo "[crafter eval ${TREE_DEPTH:+tree depth=$TREE_DEPTH branches=$BRANCHES keep=$KEEP_BRANCHES segment_turns=$SEGMENT_TURNS segment_tokens=$SEGMENT_TOKENS summary_tokens=$SUMMARY_TOKENS judge=${JUDGE_BACKEND:-openai}:${JUDGE_MODEL:-config} safe_radius=${SAFE_RADIUS:-config} key=$([ -n "${GEMINI_API_KEY:-}" ] && echo set || echo UNSET) }r$KEEP-1 n$N c$CONC ep_steps=${EP_STEPS:-10000} max_tokens=${MAX_TOKENS:-256000} respawn=${RESPAWN:-off} notebook=${NOTEBOOK:-off} workspace=${WORKSPACE:-off} extras=${EXTRAS:-off} reasoning=${REASONING:-off} relaxed=${RELAXED:-off} silent_malformed=${SILENT_MALFORMED:-off} min_tokens=${MIN_TOKENS:-off} fp32_state=${FP32_STATE:-off} weights=${WEIGHTS:-base}] skip=$SKIP $(git log --oneline -1)" | tee "$OUT/cell.txt"

rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"
SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python
export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13

if [ "$KEEP" = full ]; then
  # Same server, same sessions, no [sglang.eviction]: everything stays resident.
  sed '/^\[sglang.eviction\]/,$d' configs/crafter/inference-kve.toml > "$OUT/inference.toml"
  EVICT=()
else
  cp configs/crafter/inference-kve.toml "$OUT/inference.toml"
  EVICT=(--sglang.eviction.max-turns "$KEEP")
fi
[ -n "$FP32_STATE" ] && sed -i '/^mamba_ssm_dtype/d' "$OUT/inference.toml"
[ -n "$MAMBA_SLOTS" ] && sed -i "s/^max_mamba_cache_size = .*/max_mamba_cache_size = $MAMBA_SLOTS/" "$OUT/inference.toml"
uv run inference @ "$OUT/inference.toml" --sglang.python "$SGL_PY" "${EVICT[@]}" > "$OUT/inference.log" 2>&1 &
INFER=$!
# The front end can outlive crashed engine processes, so also give up after 20 minutes.
for _ in $(seq 120); do
  curl -sf -m 5 http://localhost:8000/health > /dev/null && break
  kill -0 $INFER 2>/dev/null || { echo "inference died"; tail -30 "$OUT/inference.log"; exit 1; }
  sleep 10
done
curl -sf -m 5 http://localhost:8000/health > /dev/null || { echo "inference not up after 20 min"; tail -30 "$OUT/inference.log"; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/"; exit 1; }
echo "inference up $(date +%T)"
if [ -n "$WEIGHTS" ]; then
  curl -sf -m 1800 -X POST http://localhost:8000/update_weights -H "Content-Type: application/json" \
    -d "{\"weight_dir\": \"$WEIGHTS\"}" || { echo "weight update from $WEIGHTS failed"; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/"; exit 1; }
  # SGLang now serves the weights dir as the model, so the eval skips its model-name check.
  echo "loaded weights $WEIGHTS $(date +%T)"
fi

( while sleep 30; do curl -s -m 5 http://localhost:8000/metrics | grep "^kvs:" > "$OUT/kvs.prom.tmp" && mv "$OUT/kvs.prom.tmp" "$OUT/kvs.prom"; done ) &
POLL=$!
DEATH=""
if [ -n "$RESPAWN" ]; then
  IFS=: read -r AT INV PEN SAFE <<< "$RESPAWN"
  DEATH="\nenv.on_death = \"respawn\"\nenv.respawn_at = \"$AT\"\nenv.death_inventory = \"$INV\"\nenv.death_penalty = ${PEN:-0}"
  [ "$SAFE" = safe ] && DEATH="$DEATH\nenv.respawn_safe = true"
fi
[ -n "$NOTEBOOK" ] && DEATH="$DEATH\nenv.taskset.notebook = true"
[ -n "$EXTRAS" ] && DEATH="$DEATH\nenv.observation_extras = true"
[ "$EXTRAS" = perception ] && DEATH="$DEATH\nenv.extras_affordances = false"
[ -n "$REASONING" ] && DEATH="$DEATH\nenv.taskset.enable_reasoning = true"
[ -n "$RELAXED" ] && DEATH="$DEATH\nenv.taskset.relaxed_reply_format = true"
[ -n "$SILENT_MALFORMED" ] && DEATH="$DEATH\nenv.announce_malformed = false"
case $WORKSPACE in
  "") ;;
  both) DEATH="$DEATH\nenv.taskset.workspace = true" ;;
  ledger) DEATH="$DEATH\nenv.taskset.workspace = true\nenv.taskset.workspace_files = false" ;;
  files) DEATH="$DEATH\nenv.taskset.workspace = true\nenv.taskset.workspace_ledger = false" ;;
  *) echo "unknown WORKSPACE $WORKSPACE"; exit 2 ;;
esac
EVAL_CONFIG=configs/crafter/eval-kve.toml
if [ -n "$TREE_DEPTH" ]; then
  EVAL_CONFIG=configs/crafter/eval-tree.toml
  # the knobs below replace the config's own lines (the second sed strips them)
  DEATH="$DEATH\nenv.tree_depth = $TREE_DEPTH\nenv.branches = $BRANCHES\nenv.keep = $KEEP_BRANCHES\nenv.max_concurrent_agents = $((BRANCHES * KEEP_BRANCHES))\nenv.segment_turns = $SEGMENT_TURNS\nenv.segment_tokens = $SEGMENT_TOKENS\nenv.summary_max_tokens = $SUMMARY_TOKENS${JUDGE_MODEL:+\nenv.judge.model = \"$JUDGE_MODEL\"}${CUT_ON_DEATH:+\nenv.cut_on_death = true}${JUDGE_MODE:+\nenv.judge_mode = \"$JUDGE_MODE\"}"
fi
sed -e "s|^env.max_episode_steps = .*|&\nenv.episode_log = \"$OUT/episodes.jsonl\"$DEATH|" \
    ${TREE_DEPTH:+-e '/^env\.tree_depth/d; /^env\.branches/d; /^env\.keep/d; /^env\.max_concurrent_agents/d; /^env\.segment_turns/d; /^env\.segment_tokens/d; /^env\.summary_max_tokens/d; /^env\.cut_on_death/d; /^env\.on_death/d; /^env\.respawn_at/d; /^env\.respawn_safe/d; /^env\.death_inventory/d; /^env\.death_penalty/d'} \
    ${JUDGE_MODE:+-e '/^env\.judge_mode/d'} \
    "$EVAL_CONFIG" > "$OUT/eval.toml"
[ -n "$MAX_TOKENS" ] && sed -i "s/^env.player.max_total_tokens = .*/env.player.max_total_tokens = $MAX_TOKENS/" "$OUT/eval.toml"
# the strip above removes the config's respawn_safe_radius line too (prefix match): re-add it after respawn_safe
[ -n "$SAFE_RADIUS" ] && sed -i -e "/^env.respawn_safe_radius = /d" -e "s|^env.respawn_safe = true|&\nenv.respawn_safe_radius = $SAFE_RADIUS|" "$OUT/eval.toml"
[ -n "$JUDGE_BACKEND" ] && sed -i "s|^env.judge.model = .*|&\nenv.judge.backend = \"$JUDGE_BACKEND\"|" "$OUT/eval.toml"
[ -n "$EP_STEPS" ] && sed -i "s/^env.max_episode_steps = .*/env.max_episode_steps = $EP_STEPS/" "$OUT/eval.toml"
[ -n "$MIN_TOKENS" ] && sed -i "s|^\[sampling\]$|[sampling]\nmin_tokens = $MIN_TOKENS|" "$OUT/eval.toml"
uv run eval @ "$OUT/eval.toml" --clean --output-dir "$OUT/run" --dashboard false \
  ${WEIGHTS:+--client.skip-model-check true} \
  --select.skip "$SKIP" --select.limit "$N" --concurrency.min-inflight "${CONC_MIN:-$CONC}" --concurrency.max-inflight "$CONC" \
  --concurrency.initial-inflight "$CONC" > "$OUT/eval.log" 2>&1
echo "eval exit $? $(date +%T)"

kill $POLL 2>/dev/null; echo "== kvs counters"; cat "$OUT/kvs.prom" 2>/dev/null
echo "== episodes"; $VENV/bin/python scripts/crafter/report_episodes.py "$OUT/episodes.jsonl" | tee "$OUT/summary.txt"
[ -n "$TREE_DEPTH" ] && { echo "== judge cost"; $VENV/bin/python scripts/crafter/judge_cost.py "$OUT" --append-ledger | tee -a "$OUT/summary.txt"; }
kill $INFER 2>/dev/null; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null
exit 0  # the outcome is in summary.txt; pkill's no-match status must not fail the job
