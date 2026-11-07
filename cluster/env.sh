# Sourced by cluster/setup.sh (login node) and cluster/job.sh (jobs): where this cluster keeps the
# KV-Streams stack and how jobs reach the outside world. Everything lives under $KVS_ROOT on scratch:
#   $KVS_ROOT/prime-rl       this repo (cloned and pulled by cluv; deps/ submodules)
#   $KVS_ROOT/prime-rl/.venv trainer/orchestrator venv (uv.lock + extras + craftax_env)
#   $KVS_ROOT/sglang         the SGLang fork (kv-eviction-dp) and its own venv
#   $KVS_ROOT/hf             Hugging Face cache (models are downloaded by setup.sh)
#   $KVS_ROOT/rl             run outputs (OUT_ROOT of rl.sbatch)
#   $KVS_ROOT/results        cluv results_path: only the Slurm .out files, synced back by `cluv sync`

KVS_CLUSTER=${CC_CLUSTER:-${SLURM_CLUSTER_NAME:-$(hostname -f 2>/dev/null)}}
case "$KVS_CLUSTER" in *mila*) KVS_CLUSTER=mila ;; esac
export KVS_CLUSTER
export KVS_ROOT=${KVS_ROOT:-$SCRATCH/kv-race/cluv}
export REPO=${REPO:-$KVS_ROOT/prime-rl}
export VENV=${VENV:-$REPO/.venv}
export SGLANG_DIR=${SGLANG_DIR:-$KVS_ROOT/sglang}
export SGLANG_VENV=${SGLANG_VENV:-$SGLANG_DIR/.venv}
export OUT_ROOT=${OUT_ROOT:-$KVS_ROOT/rl}
export HF_HOME=$KVS_ROOT/hf
# JIT and compile caches off $HOME (small, and over quota on nibi).
export XDG_CACHE_HOME=$KVS_ROOT/cache TRITON_CACHE_DIR=$KVS_ROOT/cache/triton
export FLASHINFER_WORKSPACE_BASE=$KVS_ROOT/cache TORCHINDUCTOR_CACHE_DIR=$KVS_ROOT/cache/inductor
export UV_CACHE_DIR=${UV_CACHE_DIR:-$KVS_ROOT/uv-cache} UV_LINK_MODE=copy
export PATH=$HOME/.local/bin:$PATH

if [ -n "${SLURM_JOB_ID:-}" ]; then
  export HF_HUB_OFFLINE=1 UV_OFFLINE=1 WANDB_MODE=${WANDB_MODE:-offline}
  # Compute nodes without internet reach the judge API through the cluster's proxy.
  case "$KVS_CLUSTER" in
    rorqual | narval | trillium)
      type module > /dev/null 2>&1 || . /etc/profile
      module load httpproxy
      ;;
  esac
  # The orchestrator talks to its inference servers over localhost: never through a proxy.
  export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}" NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,$NO_PROXY}"
fi
