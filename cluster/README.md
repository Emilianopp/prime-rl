# Running KV-Streams on Mila + DRAC (cluv)

Develop on Mila or locally, push the branch, then build and submit everywhere with
[cluv](https://mila-iqia.github.io/cluv/). Run every cluv command from this directory (`cluster/`).

| Cluster | Node used | Compute-node internet | Notes |
|---|---|---|---|
| fir | 4x H100 80GB, 48 CPU, 1 TB | yes | |
| rorqual | 4x H100 80GB, 64 CPU, 512 GB | no: `httpproxy` module (env.sh) | |
| nibi | 4 of 8x H100 80GB | yes | `$HOME` over quota: cluv cannot update uv there until it is cleaned |
| mila | 4x A100L 80GB, short-unkillable | yes | 3 h cap, one such job per user |

Only 80 GB GPUs (H100, A100L), so narval (A100 40GB) is not configured. Accounts: `rrg-bengioy-ad`
first, `def-bengioy` second. Each submission
goes to both, and the first to start wins.

## Once per cluster

1. Log in, so that SSH masters exist: `~/.claude/skills/cluster-auth/scripts/auth.sh fir nibi rorqual mila`.
2. Run `cluv sync <cluster>`. This installs uv, clones this repo to `$SCRATCH/kv-race/cluv/prime-rl`, and checks out your branch.
3. Copy the gitignored `.env` (`GEMINI_API_KEY`) to `$SCRATCH/kv-race/cluv/prime-rl/.env`.
4. On the login node, run `bash cluster/setup.sh` from the clone. It takes about 20-40 min the first time.
   - It builds the prime-rl venv (from `uv.lock` + extras + craftax_env).
   - It builds the SGLang fork venv (`kv-eviction-dp`, pinned by `sglang-requirements.txt`).
   - It downloads Qwen3.5-4B into `$SCRATCH/kv-race/cluv/hf`.

## Day to day

```bash
git push                                  # cluv deploys whatever is pushed
cluv sync                                 # pull on every logged-in cluster (+ fetch Slurm logs into ./logs)
cluv run fir -- bash setup.sh             # only after uv.lock / submodules / SGLang pins change
cluv submit fir --time=1:00:00 -- bash smoke.sh          # 2-step smoke test of the full stack
cluv submit first --time=12:00:00 -- env CONFIG=configs/craftax/rl-qv-mem.toml ENVS=craftax_env \
    bash ../scripts/kv_streams/rl.sbatch                 # race all clusters, keep the first to start
cluv status                               # queues + GPU availability everywhere
cluv run fir -- env CONFIG=configs/craftax/rl-qv-grave.toml BUDGET=30 bash rl_chain.sh
                                          # judge-RL chain of 3 h blocks on one cluster (games and weights carry over)
```

- Jobs run `job.sh <command>` from `cluster/`, with `env.sh` loaded. `env.sh` sets the paths, caches,
  offline flags and proxy for that cluster.
- Run outputs go to `$SCRATCH/kv-race/cluv/rl/<config>/<jobid>`.
- Only the Slurm `.out` (under `results/`) comes back with `cluv sync`.
- `submit` refuses uncommitted changes, and it syncs (pulls) before submitting.
- A queued job runs whatever the clone holds when it starts. `job.sh` warns if that differs from
  the submitted commit.
- The Mila clone (`$SCRATCH/kv-race/cluv`) is separate from the hand-managed
  `$SCRATCH/kv-race/prime-rl-kvs` + `sglang` checkouts, and doesn't touch them.
