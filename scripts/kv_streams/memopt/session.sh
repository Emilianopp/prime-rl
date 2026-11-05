#!/bin/bash
# One unattended memopt session on the login node: take an interactive short-unkillable
# allocation, start the server step, run the given rounds one after another, check
# remembered positions against the map, then give the allocation back.
#
#   nohup bash scripts/kv_streams/memopt/session.sh OUT_ROOT VARIANTS_DIR[:N] ... > session.log 2>&1 &
set -u
K=$SCRATCH/kv-race R=${REPO:-$SCRATCH/kv-race/prl-memopt-wt} OUT=${1:?out root}; shift
mkdir -p "$OUT"
salloc --no-shell -p short-unkillable --gres=gpu:a100l:4 -c 64 --mem=512G -t 3:00:00 -J memopt-s > "$OUT/salloc.log" 2>&1
J=$(grep -o "Granted job allocation [0-9]*" "$OUT/salloc.log" | grep -o "[0-9]*$")
[ -n "$J" ] || { echo "no allocation"; cat "$OUT/salloc.log"; exit 1; }
echo "allocation $J $(date +%T)"
REPO=$R VENV=$K/prime-rl-kvs/.venv nohup srun --jobid=$J --overlap -N1 -n1 bash "$R/scripts/kv_streams/memopt/serve.sh" > "$OUT/serve.log" 2>&1 < /dev/null &
for spec in "$@"; do
  vdir=${spec%%:*} n=${spec##*:}; [ "$n" = "$spec" ] && n=16
  name=$(basename "$vdir")
  N=$n REPO=$R JUDGE_CAP=${JUDGE_CAP:-3} srun --jobid=$J --overlap -N1 -n1 bash -c \
    "for i in \$(seq 120); do curl -sf -m 5 localhost:8000/health >/dev/null && break; sleep 10; done; bash $R/scripts/kv_streams/memopt/round.sh $vdir $OUT/$name" \
    > "$OUT/$name.log" 2>&1 < /dev/null
  ( cd "$R/scripts/kv_streams/memopt" && PYTHONPATH=$R/deps/verifiers:$R/environments/craftax_env:$K/pydeps/craftax-py JAX_PLATFORMS=cpu \
    "$K/prime-rl-kvs/.venv/bin/python" check_positions.py "$OUT/$name" ) > "$OUT/$name.positions.txt" 2>&1
done
scancel "$J"; echo "done $(date +%T)"
