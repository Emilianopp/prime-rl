#!/bin/bash
# Build (or update) the KV-Streams stack on this cluster's login node; idempotent, so rerun it after
# every pull that changes uv.lock, a submodule, sglang-requirements.txt or SGLANG_REF.
#
#   bash cluster/setup.sh                 # from the cluv clone ($KVS_ROOT/prime-rl)
#   cluv run fir -- bash setup.sh         # same, through cluv (pulls first)
#
# Steps: deps/ submodules -> prime-rl venv (uv.lock, extras gpu/flash-attn/kernels) + craftax_env
# (constrained to the venv's versions) -> SGLang fork at SGLANG_REF in its own venv (pinned list from
# the Mila venv; CUDA 13.0 toolchain from pip) -> model weights into $HF_HOME. Login nodes have
# internet on every cluster; compute nodes may not, so everything a job needs is fetched here.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/env.sh"
SGLANG_REF=${SGLANG_REF:-60ccc167beeb44cd1367292cb3972d89e697864b}  # kv-eviction-dp
MODELS=${MODELS:-Qwen/Qwen3.5-4B}
mkdir -p "$KVS_ROOT" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$HF_HOME"
log() { echo "[setup $KVS_CLUSTER $(date +%T)] $*"; }
# uv itself is cluv's: `cluv sync` installs or updates it to the local machine's version.
command -v uv > /dev/null || { echo "no uv: run \`cluv sync\` first"; exit 1; }
log "$(uv --version)"

log "submodules"
git -C "$REPO" submodule sync -q
# All submodules are public: clone over https (not every cluster has a GitHub ssh key).
git -C "$REPO" -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --recursive

log "prime-rl venv"
cd "$REPO"
# mooncake (multi-node KV transfer only) has no wheel for DRAC's glibc 2.34; single-node jobs never import it.
uv sync --frozen --extra gpu --extra flash-attn --extra kernels --no-install-package mooncake-transfer-engine
uv pip freeze --python "$VENV/bin/python" | grep -v '^-e' > "$KVS_ROOT/prl-constraints.txt"
uv pip install --python "$VENV/bin/python" -c "$KVS_ROOT/prl-constraints.txt" -e environments/craftax_env

log "sglang fork @ ${SGLANG_REF:0:10}"
[ -d "$SGLANG_DIR/.git" ] || git clone --filter=blob:none https://github.com/Emilianopp/sglang.git "$SGLANG_DIR"
git -C "$SGLANG_DIR" fetch -q origin
git -C "$SGLANG_DIR" checkout -q "$SGLANG_REF"
# --no-config: never pick up prime-rl's [tool.uv] overrides (they pin a different cutlass-dsl, etc.).
[ -x "$SGLANG_VENV/bin/python" ] || uv venv --no-config -q -p 3.12 "$SGLANG_VENV"
uv pip install --no-config --python "$SGLANG_VENV/bin/python" -r "$HERE/sglang-requirements.txt"
SGLANG_BUILD_RUST_EXTS=none uv pip install --no-config --python "$SGLANG_VENV/bin/python" --no-deps -e "$SGLANG_DIR/python"
# flashinfer JIT compiles against the pip CUDA 13.0 toolchain: CUDA_HOME is this dir (rl.sbatch sets it).
CU=$("$SGLANG_VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13
ln -sfn lib "$CU/lib64"
ln -sfn libcudart.so.13 "$CU/lib/libcudart.so"

log "models: $MODELS"
for m in $MODELS; do "$VENV/bin/hf" download -q "$m" > /dev/null; done

log "check"
"$VENV/bin/python" -c "import prime_rl, verifiers, renderers, craftax_env, craftax, jax, torch, flash_attn; print('prime-rl venv ok, torch', torch.__version__)"
"$SGLANG_VENV/bin/python" -c "import sglang, torch, flashinfer; print('sglang venv ok, torch', torch.__version__)"
[ -f "$REPO/.env" ] || log "WARNING: no $REPO/.env (GEMINI_API_KEY for the craftax judge): copy it in by hand"
log "done: $(git -C "$REPO" log --oneline -1)"
