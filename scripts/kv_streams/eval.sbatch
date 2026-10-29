#!/bin/bash
#SBATCH --job-name=kvs-eval
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
# Inference-only eval of an env under KV-Streams: SGLang (4 data-parallel engines,
# sessions when evicting) serving configs/$ENV/inference.toml + `uv run eval` of
# configs/$ENV/eval.toml with the renderer client, then the episode summary.
#
#   ENV=alfworld KEEP=full sbatch eval.sbatch
#
# Env: ENV (a configs/ dir with inference.toml + eval.toml whose env writes an
# episode_log: alfworld, textworld), KEEP (turn window; default: the inference
# config's; "full": no eviction, no sessions), DATA (the taskset's dataset_path), SPLIT (the
# taskset's split; default: the eval config's), N (episodes, default all), CONC
# (in-flight episodes, default 64), SET ("key = value" overrides of the eval
# source, separated by ";", e.g. "env.max_episode_steps = 400"), REPO (prime-rl checkout, deps/ submodules
# checked out), VENV, SGLANG_VENV, OUT_ROOT.
set -u
ENV=${ENV:?set ENV, e.g. alfworld} KEEP=${KEEP:-} DATA=${DATA:?set DATA, the dataset_path} SPLIT=${SPLIT:-} N=${N:-1000000} CONC=${CONC:-64}
REPO=${REPO:-$SCRATCH/kv-race/prime-rl-kvs}
VENV=${VENV:-$REPO/.venv}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
OUT=${OUT_ROOT:-$SCRATCH/kv-race/${ENV}_eval}/${SPLIT:-default}_r${KEEP:-default}_c${CONC}/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"

export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false VLLM_API_KEY=EMPTY
export UV_PROJECT_ENVIRONMENT=$VENV
export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src:$REPO/environments/${ENV}_env:$REPO/deps/verifiers:$REPO/deps/renderers${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
echo "[$ENV eval split=${SPLIT:-default} r${KEEP:-default} c$CONC] $(git log --oneline -1)" | tee "$OUT/cell.txt"

rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"
SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python
export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13

case $KEEP in
  full) sed '/^\[sglang.eviction\]/,$d' configs/$ENV/inference.toml > "$OUT/inference.toml" ;;
  "") cp configs/$ENV/inference.toml "$OUT/inference.toml" ;;
  *) sed "s/^max_turns = .*/max_turns = $KEEP/" configs/$ENV/inference.toml > "$OUT/inference.toml" ;;
esac
uv run inference @ "$OUT/inference.toml" --sglang.python "$SGL_PY" > "$OUT/inference.log" 2>&1 &
INFER=$!
for _ in $(seq 120); do  # the front end can outlive crashed engines: give up after 20 minutes
  curl -sf -m 5 http://localhost:8000/health > /dev/null && break
  kill -0 $INFER 2>/dev/null || { echo "inference died"; tail -30 "$OUT/inference.log"; exit 1; }
  sleep 10
done
curl -sf -m 5 http://localhost:8000/health > /dev/null || { echo "inference not up after 20 min"; tail -30 "$OUT/inference.log"; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/"; exit 1; }
echo "inference up $(date +%T)"

sed -e "s|^env.taskset.dataset_path = .*|env.taskset.dataset_path = \"$DATA\"\nenv.episode_log = \"$OUT/episodes.jsonl\"|" \
    ${SPLIT:+-e "s|^env.taskset.split = .*|env.taskset.split = \"$SPLIT\"|"} \
    configs/$ENV/eval.toml > "$OUT/eval.toml"
IFS=';' read -ra OVERRIDES <<< "${SET:-}"
for kv in "${OVERRIDES[@]}"; do
  kv=$(echo "$kv" | sed 's/^ *//; s/ *$//'); key=${kv%% =*}
  [ -z "$kv" ] && continue
  if grep -q "^$key = " "$OUT/eval.toml"; then
    sed -i "s|^$key = .*|$kv|" "$OUT/eval.toml"
  else
    sed -i "s|^env.episode_log = .*|&\n$kv|" "$OUT/eval.toml"
  fi
done
uv run eval @ "$OUT/eval.toml" --clean --output-dir "$OUT/run" --dashboard false --select.limit "$N" \
  --concurrency.min-inflight "$CONC" --concurrency.max-inflight "$CONC" --concurrency.initial-inflight "$CONC" > "$OUT/eval.log" 2>&1
echo "eval exit $? $(date +%T)"

$VENV/bin/python scripts/kv_streams/summarize_eval.py "$OUT/episodes.jsonl" | tee "$OUT/summary.txt"
kill $INFER 2>/dev/null; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null
exit 0  # the outcome is in summary.txt; pkill's no-match status must not fail the job
