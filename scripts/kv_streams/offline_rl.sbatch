#!/bin/bash
#SBATCH --job-name=offline-rl
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
# Off-policy RL on stored rollouts: an RL trainer on every GPU, fed by offline_rl.py
# (stored traces + per-turn credit from a scorer) instead of an orchestrator; no
# inference. The weights of the last step stay in $OUT/run/broadcasts/step_$STEPS.
#
#   STREAMS='<run>/run/*/monitors/file/traces/stream/*.zst' SCORER=craftax_env.turn_rewards:turn_rewards \
#     SCORER_ARGS='{"max_actions": 4}' STEPS=10 sbatch offline_rl.sbatch
#
# Env: STREAMS, SCORER, SCORER_ARGS, STEPS, BATCH (segments per step, 32), LR (1e-6),
# SEQ_LEN (131072: one segment per micro batch), MODEL, ENVS (environments/ packages on
# PYTHONPATH), EXTRA_PY (more PYTHONPATH, e.g. the Craftax overlay), REPO, VENV,
# VERIFIERS / RENDERERS (default $REPO/deps/...), OUT_ROOT.
set -u
STREAMS=${STREAMS:?set STREAMS} SCORER=${SCORER:?set SCORER} SCORER_ARGS=${SCORER_ARGS:-'{}'} STEPS=${STEPS:-10}
BATCH=${BATCH:-32} LR=${LR:-1e-6} SEQ_LEN=${SEQ_LEN:-131072} MODEL=${MODEL:-Qwen/Qwen3.5-4B}
ENVS=${ENVS:-craftax_env} EXTRA_PY=${EXTRA_PY:-}
REPO=${REPO:-$SCRATCH/kv-race/prime-rl-kvs} VENV=${VENV:-$REPO/.venv}
VERIFIERS=${VERIFIERS:-$REPO/deps/verifiers} RENDERERS=${RENDERERS:-$REPO/deps/renderers}
OUT=${OUT_ROOT:-$SCRATCH/kv-race/offline_rl}/${SLURM_JOB_ID:-local}
GPUS=$(nvidia-smi -L | wc -l)
mkdir -p "$OUT/run"
echo "[offline rl steps=$STEPS batch=$BATCH lr=$LR seq=$SEQ_LEN gpus=$GPUS] $SCORER $SCORER_ARGS $STREAMS" | tee "$OUT/cell.txt"

export PATH=$VENV/bin:$HOME/.local/bin:$PATH WANDB_MODE=${WANDB_MODE:-offline} TOKENIZERS_PARALLELISM=false JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1
export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src$(for e in $ENVS; do printf ":%s" "$REPO/environments/$e"; done):$VERIFIERS:$RENDERERS${EXTRA_PY:+:$EXTRA_PY}

cat > "$OUT/trainer.toml" <<EOF
max_steps = $STEPS
output_dir = "$OUT/run"

[model]
name = "$MODEL"
seq_len = $SEQ_LEN

[optim]
lr = $LR
weight_decay = 0.0
max_norm = 1.0

[rollout_transport]
type = "filesystem"

[weight_broadcast]
type = "filesystem"
EOF

"$VENV/bin/python" "$REPO/scripts/kv_streams/offline_rl.py" "$STREAMS" --output-dir "$OUT/run" \
  --scorer "$SCORER" --scorer-args "$SCORER_ARGS" --model "$MODEL" --workers "$GPUS" \
  --batch-size "$BATCH" --steps "$STEPS" --seq-len "$SEQ_LEN" > "$OUT/feeder.log" 2>&1 &
FEEDER=$!

# From a neutral cwd: torchrun -m puts the cwd on sys.path.
cd "$SLURM_TMPDIR"
torchrun --role=trainer --rdzv-endpoint=localhost:29517 --rdzv-id=$RANDOM --log-dir="$OUT/torchrun" \
  --local-ranks-filter=0 --redirect=3 --tee=3 --nproc-per-node="$GPUS" \
  -m prime_rl.trainer.rl.train @ "$OUT/trainer.toml" > "$OUT/trainer.log" 2>&1
echo "trainer exit $?"
wait $FEEDER; echo "feeder exit $?"

strip() { sed 's/\x1b\[[0-9;]*m//g'; }
echo "== trainer steps"; grep -a -E "Step [0-9]+ \|" "$OUT/trainer.log" | strip | cut -c1-300 | tee "$OUT/trainer_steps.txt"
echo "== feeder"; tail -n 20 "$OUT/feeder.log"
ls "$OUT/run/broadcasts"
