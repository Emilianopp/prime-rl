#!/bin/bash
#SBATCH --job-name=kvs-rl2n
#SBATCH --partition=long
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3:00:00
#SBATCH --output=%x-%j.out
# rl.sbatch on two 4-GPU nodes: the trainer (all 4 GPUs, FSDP), the orchestrator and the
# env servers on the first node; one SGLang server with 4 data-parallel engines (TP 1) on
# the second. No router: SGLang's DP controller keeps each streaming session on its
# engine. `rl --dry-run` writes the sub-configs with the policy client pointed at the
# second node, then each process starts as rl_local starts it. Weights cross nodes
# through the filesystem broadcast on $SCRATCH.
#
#   CONFIG=configs/craftax/rl-qv.toml ENVS=craftax_env \
#     ARGS='--orchestrator.train.algo.log-path $OUT/qv.jsonl' sbatch scripts/kv_streams/rl2node.sbatch
#
# Env as rl.sbatch: CONFIG, STEPS, ENVS, ARGS, REPO, VENV, SGLANG_VENV, OUT_ROOT; with
# craftax_env in ENVS the game package dir CRAFTAX_PY goes on PYTHONPATH.
set -u
CONFIG=${CONFIG:?set CONFIG} STEPS=${STEPS:-} ENVS=${ENVS:-textworld_env} ARGS=${ARGS:-}
REPO=${REPO:-$SCRATCH/kv-race/prime-rl-kvs}
VENV=${VENV:-$REPO/.venv}
SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
OUT=${OUT_ROOT:-$SCRATCH/kv-race/rl}/$(basename "$CONFIG" .toml)-2node/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
INFER=${NODES[1]}
echo "[rl2node $CONFIG steps=${STEPS:-config} train=${NODES[0]} infer=$INFER] $(git -C "$REPO" log --oneline -1)" | tee "$OUT/cell.txt"

# Node-local copies on both nodes (see rl.sbatch); SLURM_TMPDIR is the same path on each.
export REPO VENV SGLANG_VENV
srun --overlap -N2 -n2 --ntasks-per-node=1 bash -c '
  rsync -a --exclude .git --exclude .venv --exclude outputs "$REPO/" "$SLURM_TMPDIR/repo/"
  rsync -a "$VENV/" "$SLURM_TMPDIR/prl-venv/"
  find "$SLURM_TMPDIR/prl-venv/bin" -maxdepth 1 -type f -exec sed -i "1s|^#!$VENV/|#!$SLURM_TMPDIR/prl-venv/|" {} +
  rsync -a "$SGLANG_VENV/" "$SLURM_TMPDIR/sgl-venv/"'
REPO=$SLURM_TMPDIR/repo VENV=$SLURM_TMPDIR/prl-venv SGL_PY=$SLURM_TMPDIR/sgl-venv/bin/python

[[ " $ENVS " == *" craftax_env "* ]] && PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}${CRAFTAX_PY:-$SCRATCH/kv-race/pydeps/craftax-py}
export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 WANDB_MODE=${WANDB_MODE:-offline} TOKENIZERS_PARALLELISM=false
export UV_PROJECT_ENVIRONMENT=$VENV
export PYTHONPATH=$REPO/src:$REPO/packages/prime-rl-configs/src$(for e in $ENVS; do printf ":%s" "$REPO/environments/$e"; done):$REPO/deps/verifiers:$REPO/deps/renderers${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
[ -f "$REPO/.env" ] && { set -a; . "$REPO/.env"; set +a; }
export CUDA_HOME=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13
CFG=$OUT/configs LOGS=$OUT/logs
export OUT PRL_ATTEMPT_CONFIG_DIR=$CFG PRL_ATTEMPT_LOG_DIR=$LOGS PRL_RUN_ID=$(uuidgen | tr -d -) PRL_RUN_NAME=$(basename "$CONFIG" .toml)
export WANDB_SHARED_MODE=1 WANDB_RUN_ID=$PRL_RUN_ID

uv run rl @ "$CONFIG" --clean --dry-run --output-dir "$OUT/run" --inference.sglang.python "$SGL_PY" \
  --deployment.gpus-per-node 8 --deployment.num-train-gpus 4 --deployment.num-infer-gpus 4 \
  --orchestrator.model.client.base-url "http://$INFER:8000/v1" \
  --orchestrator.model.client.admin-base-url "http://$INFER:8000/v1" \
  ${STEPS:+--max-steps "$STEPS"} --rollout-transport.type zmq --rollout-transport.port 5655 \
  $(eval echo "$ARGS") > "$OUT/rl.log" 2>&1 || { echo "dry run failed: see $OUT/rl.log"; exit 1; }

# The launcher's default env vars per role, as `export` lines.
role_env() { uv run python -c "
import shlex, sys
from prime_rl.utils import process as p
roles = {'common': p.DEFAULT_COMMON_ENV_VARS, 'inference': p.DEFAULT_INFERENCE_ENV_VARS, 'trainer': p.DEFAULT_TRAINER_ENV_VARS}
for k, v in {**roles['common'], **roles[sys.argv[1]]}.items():
    print(f'export {k}={shlex.quote(v)}')" "$1"; }
mkdir -p "$LOGS/trainer/torchrun"

srun --overlap -N1 -n1 -w "$INFER" --gres=gpu:l40s:4 bash -c "$(role_env inference); exec uv run inference @ $CFG/inference.json" \
  > "$LOGS/inference.log" 2>&1 &
INF=$!
for f in "$CFG"/envs/*/*.json; do
  split=$(basename "$(dirname "$f")"); mkdir -p "$LOGS/envs/$split"
  ( eval "$(role_env common)"; exec uv run env-server @ "$f" ) > "$LOGS/envs/$split/$(basename "$f" .json).log" 2>&1 &
done
( eval "$(role_env common)"; export WANDB_SHARED_LABEL=orchestrator; exec uv run orchestrator @ "$CFG/orchestrator.json" ) \
  > "$LOGS/orchestrator.log" 2>&1 &
ORCH=$!
( eval "$(role_env trainer)"; export WANDB_SHARED_LABEL=trainer
  exec uv run torchrun --role=trainer --rdzv-endpoint=localhost:$((20000 + RANDOM % 20000))--rdzv-id="$PRL_RUN_ID" \
    --log-dir="$LOGS/trainer/torchrun" --local-ranks-filter=0 --redirect=3 --tee=3 --nproc-per-node=4 \
    -m prime_rl.trainer.rl.train @ "$CFG/trainer.json" ) > "$LOGS/trainer.log" 2>&1 &
TRAIN=$!
echo "started: inference on $INFER, orchestrator + trainer + env servers on ${NODES[0]} $(date +%T)"

while kill -0 $ORCH 2>/dev/null && kill -0 $TRAIN 2>/dev/null; do
  [ -f "$OUT/STOP" ] && { echo "stop: $(cat "$OUT/STOP")"; break; }
  kill -0 $INF 2>/dev/null || { echo "inference exited"; break; }
  sleep 30
done
echo "end $(date +%T)"
kill $ORCH $TRAIN $INF $(jobs -p) 2>/dev/null; sleep 10

strip() { sed 's/\x1b\[[0-9;]*m//g'; }
echo "== trainer steps"; grep -a -E "Step [0-9]+ \|" "$LOGS/trainer.log" | strip | cut -c1-300 | tee "$OUT/trainer_steps.txt"
echo "== orchestrator steps"; grep -a -E "Step [0-9]+ \|" "$LOGS/orchestrator.log" | strip | cut -c1-300 | tee "$OUT/orchestrator_steps.txt"
pkill -u "$USER" -f "$VENV/" 2>/dev/null
exit 0  # the outcome is in the step logs
