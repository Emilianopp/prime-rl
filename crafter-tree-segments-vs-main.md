# crafter-tree-segments vs upstream main

Branch `kv-streams/crafter-tree-segments`, compared with `origin/main` at `a27783797` (2026-10-30).
Both sides use the same verifiers (`6d23351a8`) and renderers (`85b36e4`), so upstream's segments
code is installed on the branch, but the Crafter env never turns it on.

**TL;DR**
- **Segments:** main plays one long game as a chain of traces in one process. The branch makes each segment its own episode, saves the world to disk, and grows a tree: each node has several children and the best are kept.
- **Summary:** main asks for a templated `<summary>` once the token budget is used. The branch uses a notes protocol (advance warning, one retry, 32k cap, optional recall probe and loss masking).
- **Reward:** main scores achievements (minus a death penalty) for the whole episode. The branch scores each segment with an LLM judge plus the achievement delta, optionally blended with the lineage's tech tier, and the judge drives beam selection.
- **Respawn:** main can end the episode or respawn at a fixed spot. The branch adds revert, respawn at last base, safe respawn (Crafter and Craftax), death cause, lessons and cut on death, all opt-in.
- **Shared:** both start a new segment from the system prompt plus the summary in a fresh session. The model-side and server-side pieces are the same: the same verifiers and renderers versions, eviction plans, the kv_death guard and the session-rewrite refusal. Not ported: upstream's segment orchestration and `[compaction]`.

## Segments

| | Upstream main | This branch |
|---|---|---|
| Unit | One episode is a chain of traces in one process (`SegmentedInteraction`). The game object stays in memory, and the episode ends on the wall clock (`max_episode_seconds`). | One segment = one episode = one trace. The world is pickled to `state_dir`, and the next segment is a new task restored from it (`child_task`). |
| Cut | `segments.new_tokens` (tokens added after the system prompt), with `summary_tokens` kept free for the summary | `segment_turns` and/or `segment_tokens` (logical positions, evicted tokens included), or `cut_on_death` |
| Shape | One line of play per episode | A tree: each node is a saved world plus a summary, with `branches` children each |
| Next-segment prompt | System prompt + summary + current observation (new session) | Same: system prompt + summary + current observation (new session) |

## Summary (handoff)

- **Upstream:** the summary prompt is appended to the observation that crosses the budget, and the text inside `<summary>` is kept. `c2f893126` added a fixed-section template (STATUS, INVENTORY, …, PLAN). `6d23351a8` takes the summary from the reasoning when the reply is empty. A resumed run can be opened with a summary.
- **Branch:** a notes protocol (`notes_protocol`):
  - The system prompt declares the notes turn, and the policy is warned `notes_warning_turns` turns before the cut.
  - The last observation drops its "reply with an action" line and asks for `<notes>…</notes>`.
  - If the notes are under 80 characters, the policy is asked once more (`summary_retry`).
  - The notes turn is capped at `summary_max_tokens` (32k), cut down to whatever room the context has left.
  - Optional: `recall_probe` adds a scored memory-check turn before the notes. `train_on_notes = false` masks the notes and probe replies out of the loss (`trace.info["loss_mask"]["drop_nodes"]`).

## Reward and judge

- **Upstream:** the reward is achievements minus `death_penalty` per death. Segments either share the episode reward (`credit = "episode"`) or are credited alone (`stream_segments`). There is no judge.
- **Branch:** the reward is `achievement_weight` × achievement delta / 22 + `judge_weight` × judge score − `death_penalty`, per segment. The judge (Gemini over the OpenAI-compatible endpoint, or Copilot API):
  - Sees the segment's start state, an event timeline, the ground truth at the cut, the nearest resources within `judge_world_radius` (judge-only, privileged), the next tech-tree step, and the policy's notes.
  - Scores progress, plan and consistency, mixed by `judge_mix` (0.5/0.3/0.2).
  - `judge_mode = group` judges a node's siblings together in one comparative call.
  - `judge_tier_weight` blends in the lineage's cumulative tech tier, so lineages that reached higher tiers aren't dropped for one flat segment.

## Search and training

- **Upstream:** no selection. GRPO groups are episodes, or single segments under `stream_segments`.
- **Branch:**
  - **Eval:** an in-env beam (`tree_depth`, `branches`, `keep`, `keep_per_parent` as a sibling cap, `root_branches`) that checkpoints to `<state_dir>/beam/<seed>.json`, so a job chain resumes it.
  - **RL:** the orchestrator's `TreeSampler` turns finished branches into the next nodes. A GRPO group is one node's `group_size` branches, and the best `keep` per depth continue.

## Death and respawn

- **Upstream Crafter:**
  - `on_death`: end | respawn.
  - `respawn_at`: spawn | last_sleep | death_site, onto the nearest free cell.
  - `death_inventory` and `death_penalty`.
- **Branch adds** (all opt-in; the defaults behave like upstream):
  - `on_death = revert` with `revert_to` = segment_start | last_base: reload a save point.
  - `respawn_at = last_base`: the most recent of where the player last slept and where it last placed a table.
  - `respawn_safe` / `respawn_safe_radius`: the nearest cell with no zombie or skeleton within the radius.
  - `announce_death_cause`: the death line names the cause and the step.
  - `death_lesson`: a `<lesson>` written after each death, pinned in later observations.
  - `cut_on_death`.
- **Craftax:** the branch also adds `respawn_safe` / `respawn_safe_radius` (melee and ranged mobs, plus projectiles).

## Infrastructure

- **Config API:** the branch keeps `[inference.sglang] sessions = true` + `[sglang.eviction]`. Main moved to `[compaction]`.
- **Ported from main** (cherry-picked with `-x`):
  - ALFWorld env.
  - TextWorld eval configs and family filter.
  - `scripts/kv_streams/{eval,rl}.sbatch`.
  - Craftax env, eval scripts and failure-reason feedback.
  - Qwen3.5 eviction plans, with a new `eviction.model_plan` switch.
  - The `supports_kv_death` trainer guard.
  - SGLang refusal of session rewrites (`allow_session_rewrite = false`).
- **Branch-only:** the Craftax eval configs drop their `segments` lines. The eval scripts size SGLang's `data_parallel_size` to the GPUs in the job.
- **Not ported:** upstream's segments orchestration (`stream_segments`, episode credit, the wip chain), `[compaction]`, continuous Crafter via segments, and Craftax resume across jobs.

## Smoke check

Run on 2026-10-30 at `b5df81833` plus the eval-script changes. All three jobs had 0 reopens, 0 resurrected turns, 0 refused rewrites and 0 failed calls.

| Job | Run | Result |
|---|---|---|
| 11459242 (`main`, 2 GPUs) | Crafter flat: window r8-1, 32 episodes, 300 steps | 1.66 achievements per episode by step 300; 1.84–2.66 for earlier runs on the same 32 seeds. Malformed replies 0.0002 per turn. |
| 11459243 (`unkillable`, 1 GPU) | Crafter tree: depth 2, 2 roots × 4 branches, 30-turn segments | 16 segments. Depth 2 restored at game step 30. 16 of 16 judged (scores 0.47–0.65, no failures). Notes 1.8k–20k characters, no retries needed. |
| 11458957 (`main`, 2 GPUs) | Craftax continuous, thinking on, feedback on, safe radius 4 | 8 episodes, 3.88 achievements per episode. Failure reasons appear ("nothing made: needs a table within one step"). No deaths, so safe respawn was not exercised live (unit tests only). |

The flat Crafter score is at the low end of the earlier runs. With 32 episodes (standard deviation about 1.3) that is within noise, but a 256-episode rerun would settle it.
