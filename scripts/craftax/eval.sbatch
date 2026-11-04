#!/bin/bash
#SBATCH --job-name=craftax-eval
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
# Inference-only full Craftax (craftax-env, GlyphBench's glyph observations): SGLang
# (4 data-parallel engines, KV-Streams) + `uv run eval` with the renderer client, then
# the episode summary.
#
#   sbatch scripts/craftax/eval.sbatch
#
# Env: KEEP (eviction max_turns, default 50; "full": eviction off, plain SGLang), CONC
# (in-flight episodes, default PER_GPU x GPUs), PER_GPU (default 32 with eviction, which
# fills about 80% of an A100's KV pool at ~45k live tokens a trace; 6 with KEEP=full, whose
# traces grow to the 256k ceiling), N (episodes; default 3 x CONC when STEPS caps the episodes,
# else CONC: above CONC finished slots refill - without it the batch drains and only ~35% of
# decode ran at a full batch in job 11468460 - and the episodes still running at the job's end
# stop at DEADLINE, logged as ended_by=deadline and left out of the summary), DEADLINE_MARGIN (seconds before the job's end at which every episode
# stops, default 600), MULTIPLEX (episodes per env worker process, default the config's 32),
# SKIP (skip the first SKIP tasks: disjoint seeds across jobs), TURN_TOKENS (max tokens per reply, default
# 2048; also the budget the system prompt states), NO_THINKING (1: the renderer's
# non-thinking mode and GlyphBench's action-only format; text: non-thinking mode with
# GlyphBench's plain-text reasoning format, as the RL configs train), TIME_LIMIT (seconds per
# episode, default the config's 9000), STEPS (turns per episode, default 10000),
# SEGMENTS (not on this branch: its verifiers has no segments; each episode is one trace
# under a 256k token ceiling), CODE_FACTS (1: the game writes KNOWN PLACES + EVENT LOG, the notes only
# FINDINGS / PLAN / SURVIVAL; env.code_facts), EXPLORATION_AID (1: GlyphBench's [Exploration Aid] after every
# observation; env.exploration_aid), ACTION_FROM_REASONING (1: play the tag a reply's reasoning ends with
# when its content has none; env.action_from_reasoning), SAFE_RADIUS (N: respawn at a free cell with no hostile mob
# within N tiles, env.respawn_safe; with CONTINUOUS),
# CONTINUOUS (1: configs/craftax/eval-continuous.toml - respawn on death, no turn limit,
# achievement reward; STEPS then sets a turn limit again), FEEDBACK (1: env.action_feedback - the
# message says what each action did), MAX_ACTIONS (N > 1: up to N actions per reply), REPO
# (prime-rl checkout, deps/ submodules checked out), VENV, SGLANG_VENV, CRAFTAX_PY (a
# --target dir with jax[cpu] and the Craftax fork, for a venv without them), OUT_ROOT.
# TREE_DEPTH (N > 0: configs/craftax/eval-tree.toml, the frozen in-env beam over segments of
# docs/craftax-segments-judge.md; each episode is a whole beam for one seed): ROOTS (beams =
# episodes = CONC, default 2), BRANCHES (4), BEAM_KEEP (2), ROOT_BRANCHES (BRANCHES x
# BEAM_KEEP), SEGMENT_TURNS (100), SEGMENT_TOKENS (96000), MAX_TOKENS (the per-segment
# ceiling env.player.max_total_tokens: SEGMENT_TOKENS + 35072 for the notes turn, at least
# 131072; under the engine's 262144), TREE_CHUNK (levels per eval process, unset: all in one; a
# beam is one episode that keeps every segment's trace in host memory, so a long run plays
# TREE_CHUNK levels per eval and the next one resumes from the beam checkpoints); saved games
# in $OUT/state (STATE_DIR: elsewhere, e.g. one a later job resumes from). MAMBA_SLOTS (the engine's DeltaNet state slots, default 400: 3 per open
# segment suffice, and each slot left unused is ~24 MB of GPU memory the KV pool loses).
# The notes judge (with TREE_DEPTH): JUDGE_BACKEND (unset: no judge, arm A0; capi: the Copilot
# API with ~/.copilot/config.json; openai: Gemini's endpoint with ~/.gemini_key), JUDGE_MODEL
# (gpt-4.1 for capi, gemini-3.8-flash for openai), JUDGE_MODE (single; group: a node's branches
# in one comparative call), NOTES_WEIGHT (0.5), ADHERENCE_WEIGHT (0: off, arm A1; 0.25: arm C, the
# adherence judge grades how each segment followed its parent's PLAN, same JUDGE_MODE).
# The 2026-11-04 reward (docs/craftax-reward-v2.md): NOTES_JUDGE (rank: one comparative call per parent
# ranks the siblings' notes; verdict: the per-finding judge), JUDGE_WEIGHT (0.5, the rank's share of the
# notes reward), FALSE_PENALTY (0.3 per false finding), ADHERENCE_MODE (code: statuses by code, no call,
# no gate; judge), STATE_WEIGHT (1: the held tools' achievement points as the `state` term; 0: off),
# BEAM_CUMULATIVE (1: the beam ranks by the lineage's total achievement score + the segment's terms;
# 0: the segment's reward alone, as before).
set -u
# One data-parallel engine per GPU of the allocation (the config says 4; main gives 2, unkillable 1).
GPUS=${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}; [ "$GPUS" -ge 1 ] || GPUS=1
KEEP=${KEEP:-50}
PER_GPU=${PER_GPU:-$([ "$KEEP" = full ] && echo 6 || echo 32)}
CONC=${CONC:-$((GPUS * PER_GPU))}
STEPS=${STEPS:-}
N=${N:-$([ -n "$STEPS" ] && echo $((3 * CONC)) || echo "$CONC")} SKIP=${SKIP:-0} TURN_TOKENS=${TURN_TOKENS:-2048} NO_THINKING=${NO_THINKING:-}
DEADLINE_MARGIN=${DEADLINE_MARGIN:-600} MULTIPLEX=${MULTIPLEX:-}
TIME_LIMIT=${TIME_LIMIT:-} STEPS=${STEPS:-} SEGMENTS=${SEGMENTS:-} CONTINUOUS=${CONTINUOUS:-} FEEDBACK=${FEEDBACK:-} MAX_ACTIONS=${MAX_ACTIONS:-} CRAFTAX_PY=${CRAFTAX_PY:-} SAFE_RADIUS=${SAFE_RADIUS:-} CODE_FACTS=${CODE_FACTS:-} EXPLORATION_AID=${EXPLORATION_AID:-} ACTION_FROM_REASONING=${ACTION_FROM_REASONING:-}
[ -n "$SEGMENTS" ] && [ "$SEGMENTS" != off ] && { echo "SEGMENTS=$SEGMENTS: this branch's verifiers has no segments (use TREE_DEPTH)"; exit 2; }
TREE_DEPTH=${TREE_DEPTH:-} TREE_CHUNK=${TREE_CHUNK:-} MAMBA_SLOTS=${MAMBA_SLOTS:-} BRANCHES=${BRANCHES:-4} BEAM_KEEP=${BEAM_KEEP:-2} SEGMENT_TURNS=${SEGMENT_TURNS:-100} SEGMENT_TOKENS=${SEGMENT_TOKENS:-96000}
ROOT_BRANCHES=${ROOT_BRANCHES:-$((BRANCHES * BEAM_KEEP))}
MAX_TOKENS=${MAX_TOKENS:-$((SEGMENT_TOKENS + 35072 > 131072 ? SEGMENT_TOKENS + 35072 : 131072))}
MAXC=$((ROOT_BRANCHES > BRANCHES * BEAM_KEEP ? ROOT_BRANCHES : BRANCHES * BEAM_KEEP))  # a level's branches play together
if [ -n "$TREE_DEPTH" ]; then ROOTS=${ROOTS:-2}; N=$ROOTS; CONC=$ROOTS; fi
JUDGE_BACKEND=${JUDGE_BACKEND:-} JUDGE_MODE=${JUDGE_MODE:-single} NOTES_WEIGHT=${NOTES_WEIGHT:-0.5} ADHERENCE_WEIGHT=${ADHERENCE_WEIGHT:-0}
NOTES_JUDGE=${NOTES_JUDGE:-rank} JUDGE_WEIGHT=${JUDGE_WEIGHT:-0.5} FALSE_PENALTY=${FALSE_PENALTY:-0.3} ADHERENCE_MODE=${ADHERENCE_MODE:-code}
STATE_WEIGHT=${STATE_WEIGHT:-1} BEAM_CUMULATIVE=${BEAM_CUMULATIVE:-1}
JUDGE_MODEL=${JUDGE_MODEL:-$([ "$JUDGE_BACKEND" = openai ] && echo gemini-3.8-flash || echo gpt-4.1)}
[ "$JUDGE_BACKEND" = openai ] && [ -z "${GEMINI_API_KEY:-}" ] && [ -r "$HOME/.gemini_key" ] && export GEMINI_API_KEY=$(cat "$HOME/.gemini_key")
REPO=${REPO:-$SCRATCH/kv-race/prime-rl-kvs}
VENV=${VENV:-$REPO/.venv}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
OUT=${OUT_ROOT:-$SCRATCH/kv-race/craftax_eval}/r${KEEP}-1_n${N}_c${CONC}_tt${TURN_TOKENS}${NO_THINKING:+_nothink$([ "$NO_THINKING" = text ] && echo -text)}${TIME_LIMIT:+_tl$TIME_LIMIT}${STEPS:+_steps$STEPS}${SEGMENTS:+_seg-${SEGMENTS//:/-}}${CONTINUOUS:+_continuous}${FEEDBACK:+_feedback}${MAX_ACTIONS:+_act$MAX_ACTIONS}${SAFE_RADIUS:+_safe$SAFE_RADIUS}${CODE_FACTS:+_codefacts}${EXPLORATION_AID:+_aid}${ACTION_FROM_REASONING:+_actreason}${TREE_DEPTH:+_tree$TREE_DEPTH-b$BRANCHES-k$BEAM_KEEP-s$SEGMENT_TURNS}${JUDGE_BACKEND:+_judge-$JUDGE_MODE}$([ -n "$JUDGE_BACKEND" ] && [ "$ADHERENCE_WEIGHT" != 0 ] && echo _adh$ADHERENCE_WEIGHT)_skip$SKIP/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"

export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false VLLM_API_KEY=EMPTY JAX_PLATFORMS=cpu
# A Craftax step is ~1.5 ms on one thread; without this every env worker sizes XLA's CPU pool to the node.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1${XLA_FLAGS:+ $XLA_FLAGS}"
export UV_PROJECT_ENVIRONMENT=$VENV
export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src:$REPO/environments/craftax_env:$REPO/environments/crafter_env:$REPO/deps/verifiers:$REPO/deps/renderers${CRAFTAX_PY:+:$CRAFTAX_PY}${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
echo "[craftax eval gpus$GPUS r$KEEP-1 n$N c$CONC tt$TURN_TOKENS thinking=${NO_THINKING:+off}${NO_THINKING:-on} time_limit=${TIME_LIMIT:-config} steps=${STEPS:-config} segments=${SEGMENTS:-config} continuous=${CONTINUOUS:-off} feedback=${FEEDBACK:-off} max_actions=${MAX_ACTIONS:-1} safe_radius=${SAFE_RADIUS:-off} code_facts=${CODE_FACTS:-off} exploration_aid=${EXPLORATION_AID:-off} action_from_reasoning=${ACTION_FROM_REASONING:-off} tree=$([ -n "$TREE_DEPTH" ] && echo "depth$TREE_DEPTH-b$BRANCHES-k$BEAM_KEEP-r$ROOT_BRANCHES-s$SEGMENT_TURNS-t$SEGMENT_TOKENS-m$MAX_TOKENS" || echo off) judge=$([ -n "$JUDGE_BACKEND" ] && echo "$JUDGE_BACKEND:$JUDGE_MODEL/$JUDGE_MODE/$NOTES_JUDGE/w$NOTES_WEIGHT" || echo off) adherence=$([ "$ADHERENCE_WEIGHT" != 0 ] && echo "$ADHERENCE_MODE/w$ADHERENCE_WEIGHT" || echo off) state_weight=$STATE_WEIGHT beam_cumulative=$BEAM_CUMULATIVE] skip=$SKIP $(git log --oneline -1)" | tee "$OUT/cell.txt"

# jobs/craftax_eval.sbatch unpacks the venv tarball here first: re-walking ~200k files on scratch costs minutes.
[ -x "$SLURM_TMPDIR/sgl-venv/bin/python" ] || rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"
SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python
export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13

# A free port per job (PORT=... to force one): on a shared node another user's server can hold 8000, and
# the health check below would take it for ours (job 11518234: "Model ... was not found in the inference pool").
PORT=${PORT:-$($VENV/bin/python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}
if [ "$KEEP" = full ]; then
  sed '/^\[sglang.eviction\]/,$d' configs/crafter/inference-kve.toml > "$OUT/inference.toml"
  EVICT=()
else
  cp configs/crafter/inference-kve.toml "$OUT/inference.toml"
  EVICT=(--sglang.eviction.max-turns "$KEEP")
fi
sed -i "s/^data_parallel_size = .*/data_parallel_size = $GPUS/; s/^port = .*/port = $PORT/" "$OUT/inference.toml"
[ -n "$MAMBA_SLOTS" ] && sed -i "s/^max_mamba_cache_size = .*/max_mamba_cache_size = $MAMBA_SLOTS  # MAMBA_SLOTS/" "$OUT/inference.toml"
uv run inference @ "$OUT/inference.toml" --sglang.python "$SGL_PY" "${EVICT[@]}" > "$OUT/inference.log" 2>&1 &
INFER=$!
for _ in $(seq 120); do
  curl -sf -m 5 http://localhost:$PORT/health > /dev/null && break
  kill -0 $INFER 2>/dev/null || { echo "inference died"; tail -30 "$OUT/inference.log"; exit 1; }
  sleep 10
done
curl -sf -m 5 http://localhost:$PORT/health > /dev/null || { echo "inference not up after 20 min"; tail -30 "$OUT/inference.log"; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/"; exit 1; }
echo "inference up $(date +%T) on port $PORT"

( while sleep 30; do curl -s -m 5 http://localhost:$PORT/metrics | grep "^kvs:" > "$OUT/kvs.prom.tmp" && mv "$OUT/kvs.prom.tmp" "$OUT/kvs.prom"; done ) &
POLL=$!
# utilization.gpu is the share of time any kernel runs, not SM occupancy: it stayed >= 90% with 15
# of 128 episodes left (job 11468460). SM occupancy / DRAM counters (DCGM, ncu) are admin-only on
# Mila (RmProfilingAdminOnly=1), so the summary estimates HBM bandwidth from the decode log instead.
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
  --format=csv,nounits -l 30 > "$OUT/gpu.csv" 2>/dev/null &
GPUMON=$!
# CPU: load and the busiest processes (top's second frame: %CPU over the last 5 s, 100 = one core).
( while sleep 25; do echo "== $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"; top -b -n 2 -d 5 -w 200 -u "$USER" | awk '/^top -/ { f++ } f == 2' | sed -n '7,22p'; done ) > "$OUT/cpu.txt" 2>&1 &
CPUMON=$!
EVAL_TOML=configs/craftax/eval.toml
[ -n "$CONTINUOUS" ] && EVAL_TOML=configs/craftax/eval-continuous.toml
[ -n "$TREE_DEPTH" ] && EVAL_TOML=configs/craftax/eval-tree.toml
sed "s|^env.max_episode_steps = .*|&\nenv.episode_log = \"$OUT/episodes.jsonl\"|; s|localhost:8000|localhost:$PORT|g" "$EVAL_TOML" > "$OUT/eval.toml"
sed -i "s/^max_completion_tokens = .*/max_completion_tokens = $TURN_TOKENS/; s/^env.taskset.max_output_tokens = [0-9]*/env.taskset.max_output_tokens = $TURN_TOKENS/" "$OUT/eval.toml"
[ -n "$FEEDBACK" ] && sed -i "s|^env.max_episode_steps = .*|&\nenv.action_feedback = true|" "$OUT/eval.toml"
[ -n "$MAX_ACTIONS" ] && sed -i "s|^env.taskset.split = .*|&\nenv.taskset.max_actions_per_turn = $MAX_ACTIONS|" "$OUT/eval.toml"
# NO_THINKING=text: the renderer's non-thinking mode but GlyphBench's reasoning format (reason in plain
# text, then the <action> tag), as configs/craftax/kve-*.toml train; any other value: action-only too.
if [ "$NO_THINKING" = text ]; then sed -i "s/^enable_thinking = true/enable_thinking = false/" "$OUT/eval.toml"
elif [ -n "$NO_THINKING" ]; then sed -i "s/^enable_thinking = true/enable_thinking = false/; s|^env.taskset.split = .*|&\nenv.taskset.enable_reasoning = false|" "$OUT/eval.toml"; fi
[ -n "$TIME_LIMIT" ] && sed -i "s/^env.max_episode_seconds = .*/env.max_episode_seconds = $TIME_LIMIT/" "$OUT/eval.toml"
[ -n "$STEPS" ] && sed -i "s/^env.max_episode_steps = .*/env.max_episode_steps = $STEPS/" "$OUT/eval.toml"
[ -n "$MULTIPLEX" ] && sed -i "s/^serve.pool.multiplex = .*/serve.pool.multiplex = $MULTIPLEX/" "$OUT/eval.toml"
if [ -n "${SLURM_JOB_ID:-}" ] && JOB_END=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s 2>/dev/null); then
  sed -i "s|^env.max_episode_steps = .*|&\nenv.episode_deadline = $((JOB_END - DEADLINE_MARGIN))|" "$OUT/eval.toml"
  echo "episodes stop by $(date -d @$((JOB_END - DEADLINE_MARGIN)) +%T) (job ends $(date -d @$JOB_END +%T))"
fi
if [ -n "$TREE_DEPTH" ]; then
  sed -i "s/^env.tree_depth = .*/env.tree_depth = $TREE_DEPTH/; s/^env.branches = .*/env.branches = $BRANCHES/; s/^env.keep = .*/env.keep = $BEAM_KEEP/;
          s/^env.segment_turns = .*/env.segment_turns = $SEGMENT_TURNS/; s/^env.segment_tokens = .*/env.segment_tokens = $SEGMENT_TOKENS/;
          s/^env.max_concurrent_agents = .*/env.max_concurrent_agents = $MAXC/; s/^env.player.max_total_tokens = .*/env.player.max_total_tokens = $MAX_TOKENS/" "$OUT/eval.toml"
  sed -i "s|^env.max_episode_steps = .*|&\nenv.root_branches = $ROOT_BRANCHES\nenv.state_dir = \"${STATE_DIR:-$OUT/state}\"|" "$OUT/eval.toml"
  sed -i "s|^env.max_episode_steps = .*|&\nenv.adherence_weight = $ADHERENCE_WEIGHT\nenv.adherence_mode = \"$ADHERENCE_MODE\"\nenv.state_weight = $STATE_WEIGHT\nenv.beam_cumulative = $([ "$BEAM_CUMULATIVE" = 0 ] && echo false || echo true)|" "$OUT/eval.toml"
  if [ -n "$JUDGE_BACKEND" ]; then
    JUDGE_LINES="env.judge.backend = \"$JUDGE_BACKEND\"\nenv.judge.model = \"$JUDGE_MODEL\"\nenv.judge.sampling.temperature = 0.0\nenv.judge_mode = \"$JUDGE_MODE\"\nenv.notes_weight = $NOTES_WEIGHT\nenv.notes_judge = \"$NOTES_JUDGE\"\nenv.judge_weight = $JUDGE_WEIGHT\nenv.false_finding_penalty = $FALSE_PENALTY"
    [ "$JUDGE_BACKEND" = openai ] && JUDGE_LINES="$JUDGE_LINES\nenv.judge.base_url = \"https://generativelanguage.googleapis.com/v1beta/openai/\"\nenv.judge.api_key_var = \"GEMINI_API_KEY\""
    sed -i "s|^env.max_episode_steps = .*|&\n$JUDGE_LINES|" "$OUT/eval.toml"
  fi
fi
[ -n "$CODE_FACTS" ] && sed -i "s|^env.max_episode_steps = .*|&\nenv.code_facts = true|" "$OUT/eval.toml"
[ -n "$EXPLORATION_AID" ] && sed -i "s|^env.max_episode_steps = .*|&\nenv.exploration_aid = true|" "$OUT/eval.toml"
[ -n "$ACTION_FROM_REASONING" ] && sed -i "s|^env.max_episode_steps = .*|&\nenv.action_from_reasoning = true|" "$OUT/eval.toml"
[ -n "$SAFE_RADIUS" ] && sed -i "s|^env.max_episode_steps = .*|&\nenv.respawn_safe = true\nenv.respawn_safe_radius = $SAFE_RADIUS|" "$OUT/eval.toml"
EVAL_ARGS=(--dashboard false --select.skip "$SKIP" --select.limit "$N" --concurrency.min-inflight "$CONC"
  --concurrency.max-inflight "$CONC" --concurrency.initial-inflight "$CONC")
if [ -n "$TREE_DEPTH" ] && [ -n "$TREE_CHUNK" ]; then
  # A resumed run (STATE_DIR seeded by an earlier job) starts at the chunk above its shallowest beam.
  D0=$($VENV/bin/python -c 'import glob, json, sys; d = [int(json.load(open(f)).get("depth") or 0) for f in glob.glob(sys.argv[1] + "/*.json")]; c = int(sys.argv[2]); print((min(d) // c + 1) * c if d else c)' "${STATE_DIR:-$OUT/state}/beam" "$TREE_CHUNK")
  for ((d = D0; ; d += TREE_CHUNK)); do
    [ "$d" -gt "$TREE_DEPTH" ] && d=$TREE_DEPTH
    sed -i "s/^env.tree_depth = .*/env.tree_depth = $d/" "$OUT/eval.toml"
    uv run eval @ "$OUT/eval.toml" --clean --output-dir "$OUT/run-d$d" "${EVAL_ARGS[@]}" >> "$OUT/eval.log" 2>&1
    echo "eval to depth $d exit $? $(date +%T)"
    [ "$d" -ge "$TREE_DEPTH" ] && break
    [ -n "${JOB_END:-}" ] && [ "$(date +%s)" -ge $((JOB_END - DEADLINE_MARGIN)) ] && break
    # go on while some beam can go deeper: stopped at a cap, or left mid-chunk by the deadline check
    # (the others reached the goal or emptied)
    $VENV/bin/python -c 'import glob, json, sys; sys.exit(not any(b.get("done") in (None, "tree_depth") for b in map(json.load, map(open, glob.glob(sys.argv[1] + "/*.json")))))' "${STATE_DIR:-$OUT/state}/beam" || break
  done
else
  uv run eval @ "$OUT/eval.toml" --clean --output-dir "$OUT/run" "${EVAL_ARGS[@]}" > "$OUT/eval.log" 2>&1
  echo "eval exit $? $(date +%T)"
fi

kill $POLL $GPUMON $CPUMON 2>/dev/null
echo "== kvs counters"; cat "$OUT/kvs.prom" 2>/dev/null
echo "== GPU (nvidia-smi every 30 s, eval window; util = time a kernel runs, not occupancy)"
awk -F', *' 'NR > 1 && $3 != "" { u[$2] += $3; m[$2] += $5; n[$2]++ } END { for (g in n) printf "gpu%s util %.0f%%  mem %.1f GB\n", g, u[g] / n[g], m[g] / n[g] / 1024 }' "$OUT/gpu.csv" 2>/dev/null | sort
# Decode is HBM-bound: a step reads the weights, every live token's KV (8 full-attention layers x K/V
# x 4 heads x 256 x bf16 = 32 KB for Qwen3.5-4B) and reads + writes each request's DeltaNet state
# (24 layers x 32 heads x 128 x 128 x bf16 = 25 MB, twice). Step time = running / gen tok/s, so
# bytes / step time is the bandwidth the engine reaches (A100-SXM4-80GB peak 2039 GB/s); "full
# batch" is the share of decode logs with at least 90% of PER_GPU running (the rest is the drain).
echo "== SGLang decode batches (per engine: mean running requests, gen tok/s, KV pool peak, HBM estimate)"
grep "Decode batch" "$OUT/inference.log" | sed -E 's/.*(DP[0-9]+)\].*#running-req: ([0-9]+), #full token: ([0-9]+).*full token usage: ([0-9.]+).*gen throughput \(token\/s\): ([0-9.]+).*/\1 \2 \3 \4 \5/' \
  | awk -v w="${HBM_WEIGHT_GB:-8.4}" -v kvb="${KV_BYTES_PER_TOKEN:-32768}" -v sb="${STATE_BYTES_PER_REQ:-50.3e6}" -v peak="${HBM_PEAK_GBS:-2039}" -v full="$PER_GPU" '
    $1 ~ /^DP/ && $2 > 0 && $5 > 0 { d = $1; r[d] += $2; t[d] += $5; k[d] = ($4 > k[d] ? $4 : k[d]); n[d]++
      bw[d] += (w * 1e9 + $3 * kvb + $2 * sb) * $5 / $2 / 1e9; f[d] += ($2 >= 0.9 * full) }
    END { for (d in n) printf "%s running %.1f  gen %.0f tok/s  KV peak %.0f%%  HBM ~%.0f GB/s (%.0f%% of peak)  full batch %.0f%%\n",
      d, r[d] / n[d], t[d] / n[d], 100 * k[d], bw[d] / n[d], 100 * bw[d] / n[d] / peak, 100 * f[d] / n[d] }' | sort
echo "== episodes"; $VENV/bin/python scripts/craftax/summarize_episodes.py "$OUT/episodes.jsonl" | tee "$OUT/summary.txt"
echo "== forfeits (also appended to the ledger: scripts/craftax/forfeit_report.py --show)"
JAX_PLATFORMS=cpu $VENV/bin/python scripts/craftax/forfeit_report.py "$OUT" --ledger | tee "$OUT/forfeits.txt"
kill $INFER 2>/dev/null; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null
exit 0  # the outcome is in summary.txt; pkill's no-match status must not fail the job
