#!/bin/bash
# The memory-prompt search's inference server, run as one step of an interactive
# allocation (salloc --no-shell; see round.sh): node-local copies of the repo and venvs
# (the round steps reuse them), then SGLang with 4 data-parallel engines under KV-Streams
# r50-1 (configs/crafter/inference-kve.toml) serving the base model until the step is
# cancelled.
#
#   nohup srun --jobid=$J --overlap -N1 -n1 bash $REPO/scripts/kv_streams/memopt/serve.sh > $OUT/serve.log 2>&1 &
set -u
REPO=${REPO:?} VENV=${VENV:?} SGLANG_VENV=${SGLANG_VENV:-$SCRATCH/kv-race/sglang/.venv}
L=$SLURM_TMPDIR
rsync -a --exclude .git --exclude .venv --exclude outputs "$REPO/" "$L/repo/"
rsync -a "$VENV/" "$L/prl-venv/"
find "$L/prl-venv/bin" -maxdepth 1 -type f -exec sed -i "1s|^#!$VENV/|#!$L/prl-venv/|" {} +
rsync -a "$SGLANG_VENV/" "$L/sgl-venv/"
touch "$L/.copied"

export PATH=$HOME/.local/bin:$PATH UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=$L/prl-venv TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$L/repo/src:$L/repo/packages/prime-rl-configs/src:$L/repo/deps/verifiers:$L/repo/deps/renderers
export CUDA_HOME=$($L/sgl-venv/bin/python -c "import site; print(site.getsitepackages()[0])")/nvidia/cu13
cd "$L/repo"
exec uv run inference @ configs/crafter/inference-kve.toml --sglang.python "$L/sgl-venv/bin/python"
