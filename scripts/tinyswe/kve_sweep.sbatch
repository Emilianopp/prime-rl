#!/bin/bash
#SBATCH --job-name=tinyswe-kve-sweep
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
# How many tinyswe episodes one node runs at once: SGLang under KV-Streams (4 engines on
# 4 L40S, about half a 4 x A100 node) serving configs/tinyswe/inference-kve.toml, real
# sandboxes on the same node, and `uv run eval` of configs/tinyswe/eval-kve.toml at each
# concurrency level in turn. Every level runs LEVEL_MIN minutes (or until N episodes finish);
# node CPU/memory/sandboxes are sampled throughout and summarize_sweep.py reports per level:
# decode throughput, running/queued requests, KV and DeltaNet-slot use, tool calls/s, sandbox
# CPU, episode boot/grade times and rewards.
#
#   LEVELS="32 64 128 192" sbatch scripts/tinyswe/kve_sweep.sbatch
#
# Env: LEVELS (space-separated in-flight episodes), LEVEL_MIN (minutes per level, default 20),
# KEEP (eviction max_turns, default 8; "full": no eviction, no sessions), THINKING (1: thinking
# mode), STEPS (turns per episode, default the eval config's 50), GROUP (rollouts per task, default
# the eval config's: 633 tasks run out within a level at >= 128 in flight), HARNESS (leaf: FrogNano's
# Leaf tools, leaf-harness, instead of one bash block per reply), DP (engines, default 4: one per
# GPU of the allocation), CONC_MIN (adaptive concurrency: each level's value is the in-flight CEILING
# and the orchestrator's controller sizes in-flight episodes from engine KV pressure down to CONC_MIN,
# starting from CONC_INIT if set, else its own (pessimistic) derived value; unset pins every level at
# its value), EXTRA ("key = value; ..." lines added to the eval source, e.g.
# "env.memory_mb = 4096"), TINYSWE_EXTRA_TARS (podman_env.sh: more images, for SWE-smith data), MAMBA_SLOTS (DeltaNet
# slots per engine, >= 3 x in-flight episodes per engine), DATA, REPO, VENV, SGLANG_VENV, OUT_ROOT.
set -u
LEVELS=${LEVELS:-32 64 128 192} LEVEL_MIN=${LEVEL_MIN:-20} KEEP=${KEEP:-8} THINKING=${THINKING:-} STEPS=${STEPS:-} GROUP=${GROUP:-} HARNESS=${HARNESS:-} DP=${DP:-} EXTRA=${EXTRA:-} CONC_MIN=${CONC_MIN:-} CONC_INIT=${CONC_INIT:-} MAMBA_SLOTS=${MAMBA_SLOTS:-}
REPO=${REPO:-$SCRATCH/kv-race/prl-tinyswe-wt}
VENV=${VENV:-$SCRATCH/kv-race/prime-rl-kvs/.venv}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
DATA=${DATA:-$SCRATCH/podman-swe-test/tiny-swe/data/tinyswe_v1}
OUT=${OUT_ROOT:-$SCRATCH/kv-race/tinyswe_sweep}/r${KEEP}${HARNESS:+_$HARNESS}${CONC_MIN:+_adaptive$CONC_MIN}${CONC_INIT:+_init$CONC_INIT}${THINKING:+_thinking}${STEPS:+_steps$STEPS}${GROUP:+_g$GROUP}/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"

export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false VLLM_API_KEY=EMPTY
export UV_PROJECT_ENVIRONMENT=$VENV
export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src:$REPO/environments/tinyswe_env:$REPO/deps/verifiers:$REPO/deps/renderers${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
echo "[tinyswe kve sweep r$KEEP levels=$LEVELS ${LEVEL_MIN}m thinking=${THINKING:-off} steps=${STEPS:-default} group=${GROUP:-default} harness=${HARNESS:-mini-swe}] $(hostname) $(nproc) cpus" | tee "$OUT/cell.txt"
ulimit -n "$(ulimit -Hn)"
source "$REPO/scripts/tinyswe/podman_env.sh"

rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"
SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python
export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13
if [ "$KEEP" = full ]; then
  sed '/^\[sglang.eviction\]/,$d' configs/tinyswe/inference-kve.toml > "$OUT/inference.toml"
  sed -i 's/^max_model_len = .*/max_model_len = 34816/' "$OUT/inference.toml"
  EVICT=()
else
  cp configs/tinyswe/inference-kve.toml "$OUT/inference.toml"
  EVICT=(--sglang.eviction.max-turns "$KEEP")
fi
[ -n "$DP" ] && sed -i "s/^data_parallel_size = .*/data_parallel_size = $DP/" "$OUT/inference.toml"
[ -n "$MAMBA_SLOTS" ] && sed -i "s/^max_mamba_cache_size = .*/max_mamba_cache_size = $MAMBA_SLOTS/" "$OUT/inference.toml"
python3 scripts/tinyswe/node_sampler.py "$OUT/node.tsv" 10 &
SAMPLER=$!
uv run inference @ "$OUT/inference.toml" --sglang.python "$SGL_PY" "${EVICT[@]}" > "$OUT/inference.log" 2>&1 &
INFER=$!
cleanup() { kill $SAMPLER $INFER 2>/dev/null; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null; $TINYSWE_PODMAN rm -af -t 0 > /dev/null 2>&1; }
trap cleanup EXIT
for _ in $(seq 120); do  # the front end can outlive crashed engines: give up after 20 minutes
  curl -sf -m 5 http://localhost:8000/health > /dev/null && break
  kill -0 $INFER 2>/dev/null || { echo "inference died"; tail -30 "$OUT/inference.log"; exit 1; }
  sleep 10
done
curl -sf -m 5 http://localhost:8000/health > /dev/null || { echo "inference not up after 20 min"; tail -30 "$OUT/inference.log"; exit 1; }
echo "inference up $(date +%T)"
sleep 60  # an idle minute: the sampler's baseline CPU of the engines alone

N_TASKS=$(wc -l < "$DATA/index.jsonl")
printf "conc\tstart\tend\n" > "$OUT/levels.tsv"
for CONC in $LEVELS; do
  L=$OUT/c$CONC; mkdir -p "$L"
  sed -e "s|^env.taskset.dataset_path = .*|env.taskset.dataset_path = \"$DATA\"\nenv.episode_log = \"$L/episodes.jsonl\"|" \
      configs/tinyswe/eval-kve.toml > "$L/eval.toml"
  [ -n "$THINKING" ] && sed -i "s/^enable_thinking = false/enable_thinking = true/" "$L/eval.toml"
  [ -n "$STEPS" ] && sed -i "s/^env.max_episode_steps = .*/env.max_episode_steps = $STEPS/" "$L/eval.toml"
  [ -n "$GROUP" ] && sed -i "s/^group_size = .*/group_size = $GROUP/" "$L/eval.toml"
  IFS=';' read -ra KVS <<< "$EXTRA"
  for kv in "${KVS[@]}"; do kv=$(echo "$kv" | sed 's/^ *//; s/ *$//'); [ -n "$kv" ] && sed -i "s|^env.episode_log = .*|&\n$kv|" "$L/eval.toml"; done
  [ "$HARNESS" = leaf ] && sed -i -e 's/^env.player.harness.id = .*/env.player.harness.id = "leaf-harness"/' \
    -e 's|^env.episode_log = .*|&\nenv.taskset.harness = "leaf"|' "$L/eval.toml"
  [ "$KEEP" = full ] && sed -i -e '/^admin_base_url/d' -e 's/^env.player.max_total_tokens = .*/env.player.max_total_tokens = 30720/' "$L/eval.toml"
  INIT=(--concurrency.initial-inflight "$CONC")
  [ -n "$CONC_MIN" ] && INIT=(${CONC_INIT:+--concurrency.initial-inflight "$CONC_INIT"})  # adaptive: seeded, or the controller derives its start
  t0=$(date +%s)
  timeout --signal=INT --kill-after=60 "${LEVEL_MIN}m" uv run eval @ "$L/eval.toml" --clean --output-dir "$L/run" --dashboard false \
    --select.limit "$N_TASKS" --concurrency.min-inflight "${CONC_MIN:-$CONC}" --concurrency.max-inflight "$CONC" \
    "${INIT[@]}" > "$L/eval.log" 2>&1
  echo "level $CONC exit $? $(( $(date +%s) - t0 )) s, episodes $(wc -l < "$L/episodes.jsonl" 2>/dev/null || echo 0)"
  printf "%s\t%s\t%s\n" "$CONC" "$t0" "$(date +%s)" >> "$OUT/levels.tsv"
  $TINYSWE_PODMAN rm -af -t 0 > /dev/null 2>&1  # sandboxes of episodes the timeout cut off
  sleep 30
done
python3 scripts/tinyswe/summarize_sweep.py "$OUT" | tee "$OUT/summary.txt"
