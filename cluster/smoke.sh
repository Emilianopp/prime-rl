#!/bin/bash
# Smoke test of the whole stack on one node, ~30-45 min: the rl-qv-mem judge-RL config shrunk to
# 2 trainer steps of 8 short segments (12k new tokens each). Exercises the SGLang fork with KV
# eviction, the Craftax env + memory, the Gemini judge (cost capped at $0.50), the trainer and a
# filesystem weight broadcast. Pass = rl.sbatch prints 2 trainer steps and 2 orchestrator steps.
#
#   cluv submit fir --time=1:00:00 -- bash smoke.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
CONFIG=${CONFIG:-configs/craftax/rl-qv-mem.toml}
OUT_ROOT=$KVS_ROOT/smoke
mkdir -p "$OUT_ROOT"
SMOKE=$OUT_ROOT/rl-qv-mem-smoke.toml
sed -E \
  -e 's/^seq_len = [0-9]+/seq_len = 16384/' \
  -e 's/^batch_size = [0-9]+/batch_size = 8/' \
  -e 's/^(initial|max)_inflight = [0-9]+/\1_inflight = 8/' \
  -e 's/cost_cap = [0-9.]+/cost_cap = 0.5/' \
  -e 's/^env.player.segments.new_tokens = [0-9]+/env.player.segments.new_tokens = 12288/' \
  -e 's/^env.player.segments.summary_tokens = [0-9]+/env.player.segments.summary_tokens = 2048/' \
  -e 's/^env.max_episode_seconds = [0-9]+/env.max_episode_seconds = 1800/' \
  "$REPO/$CONFIG" > "$SMOKE"
for want in "seq_len = 16384" "batch_size = 8" "cost_cap = 0.5" "new_tokens = 12288"; do
  grep -q "$want" "$SMOKE" || { echo "smoke config: could not set '$want'"; exit 1; }
done
export OUT_ROOT CONFIG=$SMOKE ENVS=craftax_env STEPS=2
exec bash "$REPO/scripts/kv_streams/rl.sbatch"
