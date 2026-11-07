#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# cluv job script: `cluv submit <cluster> -- <command>` runs <command> on the allocated node, from
# the project dir (cluster/), with this cluster's environment (env.sh). Commands are plain bash,
# e.g. `env CONFIG=configs/craftax/rl-qv-mem.toml ENVS=craftax_env bash ../scripts/kv_streams/rl.sbatch`.
set -u
unset PYTHONPATH  # the venv has everything; never inherit a submit shell's PYTHONPATH
. "$SLURM_SUBMIT_DIR/env.sh"
echo "[$KVS_CLUSTER job $SLURM_JOB_ID on $(hostname)] $(git -C "$REPO" log --oneline -1) :: $*"
[ -n "${GIT_COMMIT:-}" ] && [ "$GIT_COMMIT" != "$(git -C "$REPO" rev-parse HEAD)" ] \
  && echo "WARNING: submitted at ${GIT_COMMIT:0:9}, but the clone is now at $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
exec "$@"
