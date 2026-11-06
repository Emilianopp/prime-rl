# qv chain analysis

Read-only scripts over a judge-RL chain's outputs (`$SCRATCH/kv-race/rl/<config>/<job>/`:
`qv.jsonl`, `qv_states/`, `run/*/monitors/file/traces/live/*.jsonl`). Every block replays the
same 64 train worlds from turn 0, so blocks compare policies on identical games. Run with the
prime-rl venv's python; `useful.py` also needs `PYTHONPATH=<repo>/deps/verifiers`.

| script | what |
|---|---|
| `qvstat.py QV_JSONL` | judge reward per policy version (R, dV, dB, D) and cumulative cost |
| `blockstats.py OUT_JSON DIR:LABEL ...` + `blocktable.py RL_QV_DIR` | first chain (rl-qv, blocks A-I): achievements, deaths, memory, summaries, action mix per block |
| `cmp.py LABEL:UPDATES:DIR ...` | block table across chains: reward, achievements and points at block end, deaths, forfeits, memory edits at resets vs ordinary replies, reads after a reset, cost |
| `reads.py LABEL:DIR ...` | memory reads per block and by reset number within a game |
| `useful.py OUT_TXT LABEL:DIR ...` | remembered positions vs the true map at each segment end; reward after resets with vs without a read; example memories and read episodes |
| `livestats.py NEW_LIVE A_LIVE` | early-play comparison of a running block against another block, matched by replies per world |
| `extract_rewards.py CHAIN OUT_ROOT OUT_JSON` + `plot_rewards.py JSON PNG [TITLE] [MAX_UPDATE]` | judge reward vs total achievements per block, side by side |
