#!/bin/bash
#SBATCH --job-name=qv-branch
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
# The branch test of branch_test.py, on one node: serve the base policy under KV-Streams
# (SGLang, one data-parallel engine per GPU, turn window r50-1, as in the base eval),
# sample the handoff summaries, play every branch for one segment, read the end states.
# Run `branch_test.py capture` first (CPU) and `branch_test.py judge` after (login node).
#
#   OUT=<capture dir> sbatch -p <partition> scripts/kv_streams/qv/branch.sbatch
#
# Env: OUT (the capture dir: capture.pkl in, manifest/state/run out), REPO (prime-rl
# checkout, deps/ checked out), VENV, SGLANG_VENV, CRAFTAX_PY (a --target dir with jax[cpu]
# and the Craftax fork), CONC (in-flight branches, default all).
set -u
OUT=${OUT:?set OUT, the capture dir}
REPO=${REPO:-$SCRATCH/kv-race/prl-qvjudge-wt}
VENV=${VENV:-$SCRATCH/kv-race/prime-rl-kvs/.venv}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
CRAFTAX_PY=${CRAFTAX_PY:-$SCRATCH/kv-race/pydeps/craftax-py}
GPUS=$(nvidia-smi -L | wc -l)

export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false VLLM_API_KEY=EMPTY JAX_PLATFORMS=cpu
export UV_PROJECT_ENVIRONMENT=$VENV
export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src:$REPO/environments/craftax_env:$REPO/deps/verifiers:$REPO/deps/renderers:$CRAFTAX_PY${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
echo "[qv branch test gpus=$GPUS out=$OUT] $(git log --oneline -1 2>/dev/null || echo no-git)" | tee "$OUT/cell.txt"

rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"
SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python
export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13
sed "s/^data_parallel_size = .*/data_parallel_size = $GPUS/" configs/crafter/inference-kve.toml > "$OUT/inference.toml"
uv run inference @ "$OUT/inference.toml" --sglang.python "$SGL_PY" --sglang.eviction.max-turns 50 > "$OUT/inference.log" 2>&1 &
INFER=$!
stop() { kill $INFER 2>/dev/null; pkill -u "$USER" -f "$SLURM_TMPDIR/sgl-venv/" 2>/dev/null; }
for _ in $(seq 120); do
  curl -sf -m 5 http://localhost:8000/health > /dev/null && break
  kill -0 $INFER 2>/dev/null || { echo "inference died"; tail -30 "$OUT/inference.log"; exit 1; }
  sleep 10
done
curl -sf -m 5 http://localhost:8000/health > /dev/null || { echo "inference not up after 20 min"; tail -30 "$OUT/inference.log"; stop; exit 1; }
echo "inference up $(date +%T)"

$VENV/bin/python scripts/kv_streams/qv/branch_test.py sample "$OUT" > "$OUT/sample.log" 2>&1 || { echo "sampling failed"; tail -30 "$OUT/sample.log"; stop; exit 1; }
echo "summaries sampled $(date +%T)"; tail -3 "$OUT/sample.log"
EPISODES=$(cat "$OUT/episodes.txt")
N=$(( $(echo "$EPISODES" | tr ',' '\n' | wc -l) * 6 ))
CONC=${CONC:-$N}

# The base eval's settings (eval-continuous + feedback, 4 actions, memory 4096, 4200 s),
# plus: only the test games, six variants each, resumed from their saves, one segment.
sed -e "s|^env.max_episode_steps = .*|&\nenv.episode_log = \"$OUT/episodes.jsonl\"\nenv.action_feedback = true\nenv.state_dir = \"$OUT/state\"\nenv.max_segments = 1|" \
    -e "s|^env.taskset.split = .*|&\nenv.taskset.max_actions_per_turn = 4\nenv.taskset.episodes = [$EPISODES]\nenv.taskset.variants = 6|" \
    -e "s|^env.player.harness.id = .*|&\nenv.player.memory.max_key_tokens = 4096|" \
    -e "s/^env.max_episode_seconds = .*/env.max_episode_seconds = 4200/" \
    configs/craftax/eval-continuous.toml > "$OUT/eval.toml"
uv run eval @ "$OUT/eval.toml" --clean --output-dir "$OUT/run" --dashboard false \
  --select.limit "$N" --concurrency.min-inflight "$CONC" --concurrency.max-inflight "$CONC" \
  --concurrency.initial-inflight "$CONC" > "$OUT/eval.log" 2>&1
echo "eval exit $? $(date +%T)"
stop

$VENV/bin/python scripts/kv_streams/qv/branch_test.py states "$OUT" > "$OUT/states.log" 2>&1 && echo "end states read" || tail -20 "$OUT/states.log"
$VENV/bin/python - "$OUT/episodes.jsonl" <<'PY'
import collections, json, sys
lines = [json.loads(l) for l in open(sys.argv[1])]
print(len(lines), "branches ended:", dict(collections.Counter(l.get("ended_by") for l in lines)),
      "turns p50", sorted(l["turns"] for l in lines)[len(lines) // 2] if lines else None)
PY
exit 0
