#!/bin/bash
#SBATCH --job-name=tinyswe-eval
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
# Inference-only tinyswe eval: podman + the store image, then scripts/kv_streams/eval.sbatch
# with configs/tinyswe/{inference,eval}.toml (one engine: data_parallel_size must match the GPUs).
#
#   N=32 CONC=16 sbatch scripts/tinyswe/eval.sbatch
#
# Env: N, CONC, DATA (staged task dir), REPO, VENV, plus eval.sbatch's.
set -u
export REPO=${REPO:-$SCRATCH/kv-race/prl-tinyswe-wt}
export VENV=${VENV:-$SCRATCH/kv-race/prime-rl-kvs/.venv}
export DATA=${DATA:-$SCRATCH/podman-swe-test/tiny-swe/data/tinyswe_v1}
export CONC=${CONC:-16} N=${N:-32}
source "$REPO/scripts/tinyswe/podman_env.sh"
ENV=tinyswe bash "$REPO/scripts/kv_streams/eval.sbatch"
