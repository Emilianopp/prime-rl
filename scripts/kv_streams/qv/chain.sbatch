#!/bin/bash
#SBATCH --job-name=kvs-qvchain
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
# Chained judge-RL blocks (rl.sbatch with $CONFIG, default configs/craftax/rl-qv.toml) until a dollar budget
# of judge calls is spent. Each block starts from the chain's latest weights, queues its
# successor first (afterany), caps its own judge spend at what is left of the budget, and
# runs save_weights.sh beside the run: every finished weight broadcast becomes the chain's
# latest weights and the previous ones are deleted. Weights only: the optimizer state and
# the reward baseline restart each block.
#
# Chain dir $CHAIN:
#   budget        total dollars for judge calls
#   ledgers       one qv.jsonl per line: the spend so far is the sum of their max "cost"
#   base/         the model's config + tokenizer files (no weights)
#   weights/wNNNN the latest weights (NNNN = trainer updates since the base model)
#   latest        -> weights/wNNNN
#   fails         blocks in a row that saved no new weights (2 stops the chain)
#
#   CHAIN=$SCRATCH/kv-race/rl/rl-qv/chain REPO=... VENV=... sbatch scripts/kv_streams/qv/chain.sbatch
#
# Elsewhere (the DRAC clusters, cluster/rl_chain.sh) the next block goes through a job script
# instead of this file's #SBATCH header: CHAIN_JOB (e.g. cluster/job.sh, which loads env.sh)
# and CHAIN_SBATCH (that cluster's sbatch arguments: account, GPUs, CPUs, memory, time, output).
# A config without a judge cost_cap (a scorer with no API calls) runs until stopped.
set -u
C=${CHAIN:?set CHAIN} REPO=${REPO:?set REPO}
HERE=$REPO/scripts/kv_streams PY=${VENV:?set VENV}/bin/python
CONFIG=${CONFIG:-configs/craftax/rl-qv.toml} NAME=$(basename "${CONFIG:-rl-qv.toml}" .toml)
OUT=${OUT_ROOT:-$SCRATCH/kv-race/rl}/$NAME/$SLURM_JOB_ID
export CONFIG
export CHAIN REPO

latest=$(readlink -f "$C/latest"); step=$(cat "$latest/.global_step")
# No new weights since the last block started: count it; two in a row ends the chain.
if [ -f "$C/last_start_step" ] && [ "$(cat "$C/last_start_step")" = "$step" ]; then
  echo $(( $(cat "$C/fails" 2>/dev/null || echo 0) + 1 )) > "$C/fails"
else
  echo 0 > "$C/fails"
fi
[ "$(cat "$C/fails")" -ge 2 ] && { echo "chain stopped: two blocks without new weights (at w$step)"; exit 0; }
echo "$step" > "$C/last_start_step"

spent=$("$PY" - "$C/ledgers" <<'EOF'
import json, sys
total = 0.0
for path in open(sys.argv[1]).read().split():
    try:
        total += max((json.loads(l)["info"].get("cost", 0) for l in open(path) if l.strip()), default=0)
    except FileNotFoundError:
        pass
print(f"{total:.4f}")
EOF
)
left=$("$PY" -c "print(f'{$(cat "$C/budget") - $spent:.4f}')")
echo "[chain block $SLURM_JOB_ID] start w$step, spent \$$spent, left \$$left $(date +%F' '%T)" | tee -a "$C/chain.log"
"$PY" -c "import sys; sys.exit(0 if $left >= 0.25 else 1)" || { echo "budget spent: chain done" | tee -a "$C/chain.log"; exit 0; }

# Older blocks' broadcasts are superseded by the saved weights.
for f in $(cat "$C/ledgers"); do rm -rf "$(dirname "$f")"/run/*/broadcasts; done

if [ -n "${CHAIN_JOB:-}" ]; then
  # shellcheck disable=SC2086  # CHAIN_SBATCH is a list of sbatch arguments
  # from the job script's dir: it loads $SLURM_SUBMIT_DIR/env.sh
  (cd "$(dirname "$CHAIN_JOB")" && sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL ${CHAIN_SBATCH:-} \
    "$CHAIN_JOB" bash "$HERE/qv/chain.sbatch")
else
  sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL --chdir="$(dirname "$OUT")" "$HERE/qv/chain.sbatch"
fi
echo "$OUT/qv.jsonl" >> "$C/ledgers"

mkdir -p "$C/cfg"
sed -E "s/cost_cap = [0-9.]+/cost_cap = $left/" "$REPO/$CONFIG" > "$C/cfg/$NAME.toml"
if grep -q "cost_cap = " "$REPO/$CONFIG"; then
  grep -q "cost_cap = $left" "$C/cfg/$NAME.toml" || { echo "could not set cost_cap"; exit 1; }
fi

export STEP0=$step
CONFIG=$C/cfg/$NAME.toml ENVS=craftax_env \
  ARGS="--model.name $latest --orchestrator.train.algo.log-path \$OUT/qv.jsonl" \
  WATCH="bash $HERE/qv/save_weights.sh" \
  bash "$HERE/rl.sbatch"
