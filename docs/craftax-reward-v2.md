# Craftax segment reward v2: code state value, rank judge, code adherence

2026-11-04 · Sujin Yun · branch `craftax-code-map` (worktree `wt-codemap`), on top of a63920c8b

Why the 2026-11-03 reward (`craftax-reward-code-facts.md`) was replaced, what the new one computes,
and how to turn each part on. The old parts stay selectable (`notes_judge = "verdict"`,
`adherence_mode = "judge"`, `beam_cumulative = false`, `state_weight = 0`).

## What the judged beam showed (w16-96k, job 11525073, 2 worlds, depth 22-24)

| | progress only | progress + judge |
| --- | --- | --- |
| best achievements, seed 42 / 43 | 20 / 14 | 13 / 11 |
| depth-2+ segments with progress 0 | most | 622 of 674 |
| levels where the keep-4 cut fell inside a progress tie | most | 41 of 44 |
| deaths per segment, kept vs dropped, at full-tie levels | 0.17 vs 1.13 | 0.75 vs 0.87 |
| pickaxe held at the cut, kept vs dropped | 0.12 vs 0.04 | 0.04 vs 0.04 |

- Progress is sparse, so the judge terms decided almost every level, and they do not predict the
  children's progress (pairwise agreement within a level: notes 0.55, adherence 0.43, on_path 0.28,
  deviation 0.33; findings 0.67, waste 0.65, feasible 0.63 are the only parts above chance).
- The continuous judge terms made the "fewer deaths" tie-break never fire, and deaths / held tools
  are what predicts the children (parent's pickaxe tier -> children's progress 0.79).
- Group-mode scores are relative inside one parent's siblings but the cut compares across parents:
  a weak parent's best child (0.96) beat a strong parent's best (0.6). The beam dropped every
  most-advanced lineage at 3 levels (seed 42's kept set fell from 13 to 11 achievements).
- `on_path` graded the treadmill: under the drop-everything respawn the critical path starts with
  `place_table` / `make_wood_pickaxe` in 688 of 757 segments (539 of the 564 with 10+ achievements).
  `deviation` was asked 112 times and answered 0 in 78. Adherence gave "attempted" (0.6) to 376 of
  716 segments.

Memory note: `judged-beam-w16-96k-diagnosis`.

## The reward of a segment

```
reward     = progress + state_weight x state + notes_weight x notes + adherence_weight x adherence
beam_score = reward + achievement score the lineage had before this segment     (beam_cumulative)
```

Defaults: `state_weight 1.0`, `notes_weight 0.5`, `adherence_weight 0` (0.25 to turn it on),
`beam_cumulative true`. The in-env beam keeps the `keep` highest `beam_score` (ties: fewer deaths,
then branch order; `crafter_env.taskset.select_children`). RL (`advantage_mode = "per_term"`)
standardizes each term within the sibling group and routes it (below); `beam_score` is not used there
(siblings share the lineage's past).

### progress (code)

Craftax's achievement reward (1 / 3 / 5 / 8 by tier) unlocked in this segment, as before.

### state (code, `world.state_value`)

The achievement points of what the player holds at the cut: pickaxe tier (wood 1, stone 2, iron 3,
diamond 6, the cumulative reward of the crafts that give them), sword tier (same), a table (1) and a
furnace (1) standing on a seen cell of the current floor. A lineage that kept its stone pickaxe
outranks one that died and holds nothing; between siblings it is the end state that differs. Logged as
`state_value` and `state_parts`. In RL give it a std floor like progress (`term_std_floor`, done in
`configs/craftax/kve-summary.toml`).

### notes (code + rank judge, `notes_judge = "rank"`)

Code, at the cut (`plan_check.check_plan`):

- `feasible`: as before (the fraction of achievement-naming subgoals whose tools and stations are
  held at the end or made by an earlier subgoal; a craft already unlocked with its tool still held is
  wasted). `None` when no subgoal names an achievement.
- `on_track` (new): 1 when the plan chains to an achievement the lineage has NOT unlocked (the
  `target`: the first such subgoal) and every subgoal before it serves it (a step of the target's
  requirement chain, `world.requirements`; a run for a material that chain still lacks; a search for
  the feature that gives one); 0.5 when a target exists but an earlier subgoal is a detour; 0 when the
  plan only repeats unlocked achievements or names none. The restart plan "wood, table, wood pickaxe"
  scores 0; "wood, table, wood pickaxe, stone pickaxe" scores 1 for a lineage without the stone pickaxe.
- `on_path` is still computed and logged, not scored.
- `plan = mean(feasible, on_track)` (`plan_track_score`; `on_track` alone when feasible is None).

Judge (`judge.CF_RANK_PROMPT`, `CFRankVerdict`; Gemini 3.8 flash at T 0 in the runs so far): ONE
comparative call per parent over its siblings' notes, in `judge_mode = "group"`. It sees the shared
start record and, per branch, the end record, the events, the facts the game writes for the next
segment (KNOWN PLACES, TURNS LOST, EVENT LOG) and the notes. No floor map, no plan check, no deviation
question. It returns a `rank` per branch (1 = the notes that would help the next segment most, a
permutation of 1..n, else the call is rejected) and, per finding, `truth` (true / false /
unsupported).

```
rank_score = (n - rank) / (n - 1)                                    best 1, worst 0
notes      = (1 - judge_weight) x plan + judge_weight x rank_score
             - false_finding_penalty x (false findings / findings)    clipped to [0, 1]; unsupported costs nothing
notes      = plan        with no siblings in one call (single mode, a group fallback, a failed call)
notes      = 0           when the notes are empty after the retry (no call)
```

`judge_weight` 0.5, `false_finding_penalty` 0.2 (0.3 in the 2026-11-04 smokes). The rank is only ever used inside the parent's group;
across parents the beam is decided by `beam_score`'s progress and state. Judge line fields: `rank`,
`rank_score`, `false_findings`, `plan`, `plan_feasible`, `plan_on_track`, `plan_on_path`, `claims`
(`[{n, truth}]`), `mode` (`group`, or `code` when ranked alone).

### adherence (code, `adherence_mode = "code"`, `judge.code_adherence_verdict`)

How the child worked on the parent's PLAN, from the digest `adherence.AdherenceTracker` already
builds: `done` when code saw the subgoal's achievement unlock, its item count or its ladder kills
reached (1.0); `attempted` when not done but the segment came within 2 tiles of the subgoal's seen
place (0.6); `unchecked` when code can check nothing about it (no achievement, item count, kills or
seen place: not scored); else `ignored` (0). Subgoals already true at the start are not scored.
Weights 0.5 / 0.3 / 0.2 (0.1 after), renormalized over the scored ones. No LLM call, no gate (the
parent's plan is shared by the siblings, so a gate was a constant per-term normalization cancels).
Still on the child's trace only.

## Where each term trains (RL, `train_on_notes = true`)

`crafter_env.taskset.request_term_routing` writes `trace.info["term_routing"] = {"notes": [node
indexes of the notes turn]}`; `prime_rl.orchestrator.algo.grpo.GRPOAlgorithm.route_terms` then gives
the standardized `notes` term to those tokens only and the other terms (progress, state, adherence)
to the action turns. Without the request (or `train_on_notes = false`) the per-term sum goes to every
trainable token, as before. Unit test: `tests/unit/orchestrator/test_advantage.py`
(`test_per_term_routing_sends_each_term_to_its_nodes`).

## Knobs

Config (`CraftaxEnvConfig`): `notes_judge` (`rank` | `verdict`), `judge_weight`,
`false_finding_penalty`, `adherence_mode` (`code` | `judge`), `adherence_weight`, `state_weight`,
`beam_cumulative`; `judge` and `judge_mode` as before. `adherence_mode = "code"` needs no judge.

Eval / beam, `scripts/craftax/eval.sbatch` (through `jobs/craftax_eval.sbatch CODE_REPO=$AGENT_RL/wt-codemap`
until the main checkout has this code): `NOTES_JUDGE` (rank), `JUDGE_WEIGHT` (0.5), `FALSE_PENALTY`
(0.2), `ADHERENCE_MODE` (code), `ADHERENCE_WEIGHT` (0), `STATE_WEIGHT` (1), `BEAM_CUMULATIVE` (1).
Adherence, state and the beam rule are written whenever `TREE_DEPTH` is set; the judge lines when
`JUDGE_BACKEND` is.

```bash
CODE_FACTS=1 ACTION_FROM_REASONING=1 TREE_DEPTH=32 BEAM_KEEP=4 ROOT_BRANCHES=16 SEGMENT_TOKENS=98304 \
JUDGE_BACKEND=openai JUDGE_MODE=group ADHERENCE_WEIGHT=0.25 CODE_REPO=$AGENT_RL/wt-codemap \
sbatch ... craftax_eval.sbatch
```

RL, `jobs/craftax.sbatch kve-summary`: `NOTES_JUDGE`, `JUDGE_WEIGHT`, `FALSE_PENALTY`,
`ADHERENCE_MODE`, `STATE_WEIGHT`, each written into the config only when set (the main checkout may
not have the fields yet).

## Tests

`environments/craftax_env/tests/test_reward_v2.py` (10): on_track targets, state value, the rank
verdict's arithmetic, the rank judge in the beam (prompt contents, ranks, false findings), the code-only
fallback, code adherence from a digest and in a beam without a judge, the cumulative beam score in
`select_children` and `play_segment`, the routing request. The older suites still pass with the old
modes pinned (`notes_judge="verdict"`, `adherence_mode="judge"`): craftax 77, crafter 58, orchestrator 17.

## The prompts (verbatim from the code at 4039d67ab)

The notes request the model answers (`PLAN_PROMPT`) now asks for **at most 2 findings** and says
"none" is a normal answer (it asked for up to 3 and "none only if nothing went wrong" until
2026-11-04, when the smokes showed every branch writing 3 and Gemini judging 42% of them false); its
opener, retry and what the next segment starts from (`summary_text`) are unchanged from
`craftax-reward-code-facts.md`. The judge's
guide `CF_GUIDE` is unchanged too. Placeholders in `{braces}` are filled by `judge.judge_fields`.

### `PLAN_PROMPT` (the notes request, since 2026-11-04)

```text
Your notes for the next segment. This reply is not a game turn: no action will be played. The game continues from exactly this state in a fresh context, which will show the game's own record of your state, the KNOWN PLACES, TURNS LOST and EVENT LOG above, so do not copy any of them. Write only what the game cannot: what you will do and why, inside <notes>...</notes>, under these three headers:

FINDINGS: at most 2 lessons, one line each, and only for something that really cost you this segment: `N. <what went wrong, with its turn or place> because <the cause> -> <the fix> (in PLAN k or SURVIVAL k)`. If TURNS LOST counts 10 or more lost turns, finding 1 explains its first (largest) line. The cause must be what really happened in this game (the event log and the game's messages show it), not a guess about the rules. Carry each fix out in the plan: in how a subgoal is done (the way you go, what you face first) or as a SURVIVAL rule, and name where at the end. Do not add a subgoal just for a fix. No general advice, and no finding for its own sake: "none" is a normal answer when nothing worth a lesson happened.

PLAN: 2 to 4 numbered subgoals in order, one per line: `N. <achievement or item count> at <place> - done when <what the HUD will show>`. Take every place from KNOWN PLACES as written there, F<floor> (row,col); for something never seen, give a direction to search. Subgoal 1 is the step that is marked "needed next" unless a vital or a danger comes first, and then say so; your fixes change how you do the subgoals, not which comes first. Plan only achievements that are not in [Achievements already unlocked], or item counts you do not hold yet, and make sure each subgoal has its tools and station from the inventory or an earlier subgoal.

SURVIVAL: one or two rules, each `if <vital or condition> then <what to do, where>`, with places from KNOWN PLACES.

An example from a different game (its places and facts are not yours, do not copy them):
<notes>
FINDINGS:
1. Turns 30-52: 14 moves blocked by water near F0 (18,30) because I kept walking west into the lake -> walk around it by the south shore (in PLAN 1)
2. Turns 60-75: zombies hit me on open grass at night far from the table -> be at the table by dusk (in SURVIVAL 2)
PLAN:
1. 2 wood from the trees near F0 (22,31), going by the south shore of the lake - done when Wood:2
2. make_stone_pickaxe at the table F0 (25,30) - done when the HUD shows a stone pickaxe
3. collect_iron at F0 (14,36) - done when Iron:1
SURVIVAL:
if drink < 4 then drink at the water F0 (27,26)
if it is night then wall in with stone next to the table F0 (25,30) and sleep
</notes>

Be specific. Reply with <notes>...</notes> only, no <action> tag.
```

### `CF_RANK_PROMPT` (one call per parent, `judge_mode = "group"`, `notes_judge = "rank"`)

```text
You are comparing the notes a language-model agent wrote at the end of a segment of a Craftax game, for the next segment. The branches below all started from the same saved world and the same notes, then each played its own segment and wrote its own notes. The game itself writes the places the agent has seen (KNOWN PLACES), the turns that changed nothing (TURNS LOST) and the segment's events (EVENT LOG) into the next segment, so the notes are worth only what they add beyond those: a plan whose subgoals are in the right order for the next achievement this lineage lacks, with real places and the materials and stations each step needs; findings whose cause the events support and whose fix the plan or a survival rule carries out; survival rules that address what actually hurt or killed the agent. Rank the branches' notes by how much they would help the next segment make progress, best first. A polished restart of steps already done ranks below a rougher plan that reaches something new. Also mark the findings that are wrong: the next segment reads them as advice from its past self. A finding is false only when it states a rule of the game the rules above contradict, or an event, place or cause the events and the record rule out. A cause the events allow but do not prove, an imprecise turn number, or a lesson you would have phrased differently is not false. Findings that say "none" are fine.

{guide}

== Where every branch started ==
{record_before}

{branches}

Give your verdict:
- rationale: two or three lines, written first: what separates the best notes from the worst.
- verdicts: one per branch, with the branch number as given:
  - rank: 1 for the best notes, 2 for the next, ... every branch a different rank.
  - findings: one entry per numbered FINDING, in order (an empty list when the FINDINGS say "none"): n and truth ("true" when the rules, the events and the record support its cause; "false" only when they contradict it: a rule that does not exist, an event or place that did not happen, a cause the events rule out; "unsupported" when you cannot tell, which costs the agent nothing).
```

### `CF_RANK_BRANCH_BLOCK` (one per branch inside `{branches}`)

```text
== Branch {branch} ==
-- the game record at the end --
{record_after}
-- what happened ({steps} turns) --
{events}
actions played: {actions}; forfeited turns: {forfeits}
-- what the game wrote for the next segment --
{facts}
-- its notes --
{notes}
```

`{facts}` is what the game writes for the next segment (KNOWN PLACES, TURNS LOST, EVENT LOG and the
exploration aid), `{events}` the segment's timeline, `{record_after}` the GAME RECORD at the cut.
Compared with the verdict judge's prompt: no floor map, no plan check, no per-finding lesson /
applied, no waste, no deviation.

### Response format (`CFRankVerdict`)

```json
{
  "rationale": "Branch 1 plans the stone pickaxe with the wood and stone it still lacks in order; branch 0 restarts at the table it already has.",
  "verdicts": [
    {"branch": 0, "rank": 2, "findings": [{"n": 1, "truth": "true"}, {"n": 2, "truth": "false"}]},
    {"branch": 1, "rank": 1, "findings": [{"n": 1, "truth": "true"}]}
  ]
}
```

`rank` must be a permutation of 1..n (else the call is rejected and every branch gets the code half
alone); `truth` is one of "true", "false", "unsupported". Code turns this into the reward (section
"notes" above): here branch 1 gets rank_score 1 and branch 0 gets rank_score 0 with one false finding
of two, so notes = 0.5 x plan + 0 - 0.3 x 0.5.

### Adherence by code: what is checked

Per PLAN subgoal of the parent, from `adherence.AdherenceTracker` (the digest the judge mode also read):
the named achievement unlocked in this segment, the item count reached (`Wood:2`), the floor's kills
reached for a ladder, the closest distance to the subgoal's seen place. No prompt: `code_adherence_verdict`.

## Smoke runs (2026-11-04)

Eval (100-turn segments, depth 3, 2 branches, keep 2, 4 worlds, Gemini rank judge, code adherence 0.25,
state 1, cumulative beam): Mila 11534874 (short-unkillable, cn-g003, from `wt-codemap`), fir 2478665
(from `wt-rw2`). Train (kve-summary, 4 steps, batch 16, 20k-token segments, depth 3, same reward):
Mila 11535178, fir 2486117. Earlier Mila attempts died on a cluster-wide Slurm task-prolog error
(cn-g010/12/13/14/20) and once on a duplicate `env.adherence_weight` the job script wrote (11534833,
fixed in jobs/craftax.sbatch and jobs/alliance/craftax.alliance.sbatch). Results go under
`$SCRATCH/factory/runs/craftax-eval/<job>` and `.../craftax-kve-summary/<job>`.

## Width 8 and the group-judge deadlock (fixed 2026-11-04)

In the 2026-11-02 fir sweep, width 8 with a group judge deadlocked (1943388: 128 in flight, every
group judged after the 1800 s timeout with 2 to 7 of 8 members, 66 segments running at the peak).
Two causes, both fixed:

- **Partial groups.** The dispatcher took one permit per member as it dispatched, so under a shrinking
  cap (the KV concurrency controller) or a burst of fresh groups a node's branches could be half in
  flight. A branch that finished waited in `finalize` for siblings the cap kept out, holding the permit
  they needed. Now a fresh group opens only when every one of its `group_size` permits is free, the
  permits are reserved on the group (`GroupState.reserved`, `Dispatcher._take_permit`), its members
  dispatch on them without competing, and a dropped group hands unused permits back. The tree sampler's
  node is only popped once the group can start.
- **Dead siblings.** A branch that failed, or whose episode was cancelled, never wrote its judge inputs,
  so its siblings waited the full timeout. Now `finalize` marks a failed branch gone and `run` marks a
  cancelled one gone (`GroupRendezvous.mark_gone`: `gone_<member>.json`); gone members count toward the
  set, the present ones are judged together at once, and a member left alone is judged alone (code half
  only under the rank judge) without a call.

Tests: `tests/unit/orchestrator/test_group_admission.py`, `environments/craftax_env/tests/test_group_gone.py`.
Not yet exercised at width 8 on a GPU: the next throughput sweep cell should be w8 (WIDTH=8).

## Not done

- Offline check of `on_track` against the 2026-11-04 run (needs the kept nodes' saved games).
- The smoke runs above: Gemini accepts the schema, ranks are permutations, the four trainer steps run with routed advantages (pending).
- The main checkout `prime-rl` is still at a63920c8b-equivalent; the first beam race keeps running from
  the frozen `wt-beam`, the v2 race from `wt-beam2` (d5e345f5d).
- A width-8 training cell with the group-admission fix.
