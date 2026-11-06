#!/bin/bash
# One round of the memory-prompt search: every variant in a folder plays the same worlds
# at once, one `uv run eval` per variant against the shared server (serve.sh), then
# score.py ranks them. Each variant is a TOML fragment of [[source]] keys appended to
# configs/craftax/memopt.toml - env.player.memory.{prompt,turn_note,summary_note} and
# env.player.segments.summary_prompt; an empty file is the default prompts.
#
#   srun --jobid=$J --overlap -N1 -n1 bash $REPO/scripts/kv_streams/memopt/round.sh VARIANTS_DIR OUT_DIR
#
# Env: N (worlds per variant, default 16), SKIP (first task, default the config's 1000),
# SECONDS_PER_EPISODE (default the config's 1800), REPO (the base config and score.py are
# read from it, so edits apply without a restart), JUDGE_CAP (dollars of Gemini memory grading
# per round, default 3). Needs serve.sh's node-local copies.
set -u
VDIR=$(readlink -f "${1:?variants dir}") OUT=$(readlink -f "${2:?out dir}")
N=${N:-16} L=$SLURM_TMPDIR R=${REPO:-$SCRATCH/kv-race/prl-memopt-wt}
[ -f "$L/.copied" ] || { echo "serve.sh has not copied the repo/venvs to $L"; exit 1; }
curl -sf -m 5 http://localhost:8000/health > /dev/null || { echo "inference server not up"; exit 1; }
export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=$L/prl-venv WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false VLLM_API_KEY=EMPTY JAX_PLATFORMS=cpu
export PYTHONPATH=$L/repo/src:$L/repo/packages/prime-rl-configs/src:$L/repo/environments/craftax_env:$L/repo/deps/verifiers:$L/repo/deps/renderers:${CRAFTAX_PY:-$SCRATCH/kv-race/pydeps/craftax-py}
cd "$L/repo"
mkdir -p "$OUT"
echo "[memopt round $(basename "$VDIR")] $(ls "$VDIR"/*.toml | wc -l) variants x $N worlds, $(date +%T)" | tee "$OUT/round.txt"

pids=()
for v in "$VDIR"/*.toml; do
  name=$(basename "$v" .toml); o=$OUT/$name; mkdir -p "$o"
  { sed "s|@OUT@|$o|g" "$R/configs/craftax/memopt.toml"; echo; cat "$v"; } > "$o/eval.toml"
  [ -n "${SECONDS_PER_EPISODE:-}" ] && sed -i "s/^env.max_episode_seconds = .*/env.max_episode_seconds = $SECONDS_PER_EPISODE/" "$o/eval.toml"
  timeout 60m uv run eval @ "$o/eval.toml" --clean --output-dir "$o/run" --dashboard false \
    --select.limit "$N" ${SKIP:+--select.skip "$SKIP"} \
    --concurrency.min-inflight "$N" --concurrency.max-inflight "$N" --concurrency.initial-inflight "$N" \
    > "$o/eval.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo "round done $(date +%T)" | tee -a "$OUT/round.txt"
"$L/prl-venv/bin/python" "$R/scripts/kv_streams/memopt/score.py" "$OUT" | tee "$OUT/score.txt"
# Gemini grades the memories (GEMINI_API_KEY from $R/.env, gitignored); JUDGE_CAP dollars per round.
[ -f "$R/.env" ] && { set -a; . "$R/.env"; set +a; }
[ -n "${GEMINI_API_KEY:-}" ] && "$L/prl-venv/bin/python" "$R/scripts/kv_streams/memopt/judge_memory.py" "$OUT" \
  --cap "${JUDGE_CAP:-3}" | tee "$OUT/judge.txt"
