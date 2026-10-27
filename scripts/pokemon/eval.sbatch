#!/bin/bash
# Pokémon FireRed eval under KV-Streams: SGLang (4 DP engines) + `uv run eval` on one node.
#   sbatch scripts/pokemon/eval.sbatch                      # r50-1 eviction, 64 episodes
#   KEEP=full N=64 CONC=64 STEPS=1000 MODEL=Qwen/Qwen3.5-4B sbatch scripts/pokemon/eval.sbatch
# Needs: the FireRed ROM + start state (POKEMON_FIRERED_ROM / POKEMON_FIRERED_STATE) and the mgba
# bindings with libmgba (MGBA_SITE, built by environments/pokemon_env/scripts/build_libmgba.sh).
#SBATCH -J pokemon-eval
#SBATCH -p short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH -t 3:00:00
#SBATCH --exclude=cn-g029
#SBATCH -o /network/scratch/e/emiliano.penaloza/kv-race/pokemon_eval/slurm-%j.out
set -u
KEEP=${KEEP:-50} N=${N:-64} CONC=${CONC:-64} STEPS=${STEPS:-1000} MODEL=${MODEL:-} SKIP=${SKIP:-0}
REPO=${REPO:-$SCRATCH/kv-race/prime-rl-kvs}
VENV=${VENV:-$SCRATCH/kv-race/prime-rl-kvs/.venv}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
MGBA_SITE=${MGBA_SITE:-$SCRATCH/pokemon/py-mgba}
export POKEMON_FIRERED_ROM=${POKEMON_FIRERED_ROM:-$SCRATCH/pokemon/continual-harness/FireRed-GBAdvance/rom.gba}
export POKEMON_FIRERED_STATE=${POKEMON_FIRERED_STATE:-$SCRATCH/pokemon/continual-harness/FireRed-GBAdvance/start.state}
OUT=${OUT_ROOT:-$SCRATCH/kv-race/pokemon_eval}/r${KEEP}-1_n${N}_c${CONC}_s${STEPS}${MODEL:+_$(basename "$MODEL")}/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false VLLM_API_KEY=EMPTY
export UV_PROJECT_ENVIRONMENT=$VENV
export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src:$REPO/environments/pokemon_env:$REPO/deps/verifiers:$REPO/deps/renderers:$MGBA_SITE${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
echo "[pokemon eval r$KEEP-1 n$N c$CONC steps$STEPS model=${MODEL:-config}] skip=$SKIP $(git log --oneline -1)" | tee "$OUT/cell.txt"
"$VENV/bin/python" -c "import mgba.core" || { echo "mgba not importable from $MGBA_SITE"; exit 1; }

rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"
SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python
export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13
cp configs/pokemon/inference.toml "$OUT/inference.toml"
EVICT=(--sglang.eviction.max-turns "$KEEP")
if [ "$KEEP" = full ]; then
  sed -i '/^\[sglang.eviction\]/,$d' "$OUT/inference.toml"
  EVICT=()
fi
[ -n "$MODEL" ] && sed -i "s|^model = .*|model = \"$MODEL\"|" "$OUT/inference.toml"
uv run inference @ "$OUT/inference.toml" --sglang.python "$SGL_PY" "${EVICT[@]}" > "$OUT/inference.log" 2>&1 &
INFER=$!
for _ in $(seq 120); do
  curl -sf -m 5 http://localhost:8000/health > /dev/null && break
  kill -0 $INFER 2>/dev/null || { echo "inference died"; tail -30 "$OUT/inference.log"; exit 1; }
  sleep 10
done
curl -sf -m 5 http://localhost:8000/health > /dev/null || { echo "inference not up after 20 min"; tail -30 "$OUT/inference.log"; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/"; exit 1; }
echo "inference up $(date +%T)"

sed -e "s|^env.max_episode_steps = .*|env.max_episode_steps = $STEPS\nenv.episode_log = \"$OUT/episodes.jsonl\"|" \
    configs/pokemon/eval.toml > "$OUT/eval.toml"
[ -n "$MODEL" ] && sed -i "s|^model = .*|model = \"$MODEL\"|" "$OUT/eval.toml"
uv run eval @ "$OUT/eval.toml" --clean --output-dir "$OUT/run" --dashboard false \
  --select.skip "$SKIP" --select.limit "$N" --concurrency.min-inflight "$CONC" --concurrency.max-inflight "$CONC" \
  --concurrency.initial-inflight "$CONC" > "$OUT/eval.log" 2>&1
echo "eval exit $? $(date +%T)"
echo "== episodes"; "$VENV/bin/python" scripts/pokemon/summarize_episodes.py "$OUT/episodes.jsonl" | tee "$OUT/summary.txt"
kill $INFER 2>/dev/null; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null
exit 0  # the outcome is in summary.txt; pkill's no-match status must not fail the job
