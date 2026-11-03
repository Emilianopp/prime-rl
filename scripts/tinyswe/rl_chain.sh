#!/bin/bash
# Submit a tinyswe RL run as N chained short-unkillable jobs (3 h each, 4 x A100): every job uses
# the same RUN name and output dir, checkpoints every step, and resumes from the latest checkpoint
# (scripts/kv_streams/rl.sbatch, RUN). The next job starts when the previous one ends, however it ends.
#
#   RUN=tinyswe-r1 bash scripts/tinyswe/rl_chain.sh 5        (env passes through: CONFIG, STEPS, CONC, ...)
set -eu
N=${1:-4}
RUN=${RUN:?set RUN, the run name shared by the chain}
cd "$(dirname "$0")/../.."
jid=$(RUN=$RUN sbatch --parsable --job-name="$RUN" scripts/tinyswe/rl.sbatch)
echo "$jid"
for _ in $(seq 2 "$N"); do
  jid=$(RUN=$RUN sbatch --parsable --job-name="$RUN" --dependency=afterany:"$jid" scripts/tinyswe/rl.sbatch)
  echo "$jid"
done
