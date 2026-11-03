#!/bin/bash
#SBATCH --job-name=tinyswe-rl
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
# tinyswe RL on one 4-GPU node: podman + the store image, then scripts/kv_streams/rl.sbatch
# with configs/tinyswe/rl.toml. The sandboxes share the node's CPUs with the engines,
# trainer and orchestrator; a sampler logs CPU, memory and live sandboxes every 10 s to
# <out>/node.tsv, and the env logs every episode to <out>/episodes.jsonl.
#
#   STEPS=10 CONC=768 sbatch scripts/tinyswe/rl.sbatch
#   FIXED=160 INDEX=index.filtered.jsonl sbatch --time=1:30:00 scripts/tinyswe/rl.sbatch
#   RUN=tinyswe-r1 bash scripts/tinyswe/rl_chain.sh 5   # 5 chained 3 h jobs, checkpoint + resume
#
# Env: CONFIG (default configs/tinyswe/rl.toml), STEPS, CONC (in-flight ceiling; the run starts
# at half of it), FIXED (pin in-flight episodes: the controller neither grows nor trims, to
# measure throughput at a given concurrency), INDEX (task list in the dataset dir, e.g. from
# filter_tasks.py), REPO, VENV, plus rl.sbatch's.
set -u
export REPO=${REPO:-$SCRATCH/kv-race/prl-tinyswe-wt}
export VENV=${VENV:-$SCRATCH/kv-race/prime-rl-kvs/.venv}
export ENVS=tinyswe_env OUT_ROOT=${OUT_ROOT:-$SCRATCH/kv-race/rl}
CONFIG=${CONFIG:-configs/tinyswe/rl.toml} CONC=${CONC:-} FIXED=${FIXED:-} INDEX=${INDEX:-}
OUT=$OUT_ROOT/tinyswe-rl/${RUN:-${SLURM_JOB_ID:-local}}
mkdir -p "$OUT"
source "$REPO/scripts/tinyswe/podman_env.sh"

# The run's config: the episode log into OUT, the task list if INDEX is set, and the in-flight
# ceiling (CONC) or pin (FIXED).
sed -e "s|^env.max_episode_steps = .*|&\nenv.episode_log = \"$OUT/episodes.jsonl\"|" \
    ${INDEX:+-e "s|^env.taskset.dataset_path = .*|&\nenv.taskset.index = \"$INDEX\"|"} \
    ${CONC:+-e "s|^max_inflight = .*|max_inflight = $CONC|" -e "s|^initial_inflight = .*|initial_inflight = $((CONC / 2))|"} \
    ${FIXED:+-e "s|^max_inflight = .*|max_inflight = $FIXED|" -e "s|^initial_inflight = .*|initial_inflight = $FIXED|" -e "s|^min_inflight = .*|min_inflight = $FIXED|"} \
    "$REPO/$CONFIG" > "$SLURM_TMPDIR/tinyswe-rl.toml"
cp "$SLURM_TMPDIR/tinyswe-rl.toml" "$OUT/config.toml"
export CONFIG=$SLURM_TMPDIR/tinyswe-rl.toml

python3 "$REPO/scripts/tinyswe/node_sampler.py" "$OUT/node.tsv" 10 &
SAMPLER=$!
bash "$REPO/scripts/kv_streams/rl.sbatch"
kill $SAMPLER 2>/dev/null
$TINYSWE_PODMAN rm -af -t 0 > /dev/null 2>&1
exit 0
