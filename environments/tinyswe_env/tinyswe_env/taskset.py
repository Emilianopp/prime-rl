"""tinyswe-env — SWE-smith-style bug-fix tasks on pure-Python repos, one sandbox per rollout.

Tasks come from the tiny-swe pipeline: a bug planted in a library function (LM Modify), validated
by the tests it breaks (FAIL_TO_PASS) and the ones it leaves alone (PASS_TO_PASS), with an issue
written from its symptoms. Every repo shares ONE image (`/repos/<name>` checkouts, `/envs/<name>`
venvs hardlinked from a package store), so a rollout's sandbox is a throwaway container of it:
copy the checkout to /work, apply and commit the bug, then serve commands over a unix socket
(cmdserver.py, the container's main process — no `podman exec` per command).

Harness: mini-swe-agent style. Each reply holds one ```bash block, run in a fresh shell in /work;
the observation is its return code and (clipped) output. The episode ends when the model runs
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, or at max_episode_steps. Then any test files the
agent changed are restored and the suite runs: reward 1 if every FAIL_TO_PASS and PASS_TO_PASS
test passes, else 0.

Harness `leaf` (taskset.harness = "leaf" with env.player.harness.id = "leaf-harness"): FrogNano's
Leaf (github.com/microsoft/FrogNano) — function calling with Read, Write, Edit, Glob and Bash, several
calls per reply, until a reply makes none (its final answer). The env runs each call in the sandbox
(cmdserver.py `leaf` op, Leaf's own output formats) and answers with one tool message per call; a
model turn is one reply however many calls it makes. Grading is the same.

Dataset layout (`dataset_path`, from stage_tasks.py): index.jsonl (one task per line: id, repo,
problem statement, suite time), tasks/<id>.json (FAIL_TO_PASS, PASS_TO_PASS), patches/<id>.diff.
SWE-smith tasks (stage_smith.py) add `image` to the row (its jyangballin/swesmith.x86_64.* image,
repo at /testbed, conda env) and `test_files` to the task: the grade runs those files and reads
pytest node ids off the -rA summary.
"""

import asyncio
import json
import os
import random
import re
import shlex
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from leaf_harness import LEAF_SYSTEM_PROMPT
from pydantic import Field

SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SYSTEM_PROMPT = f"""You are a software engineer fixing a bug in a Python library. You work in a \
Linux shell, one command at a time.

The repository is checked out at /work, with its dependencies installed: `python` and `pytest` run \
in the project's environment. There is no network access.

Every reply has two parts: a short THOUGHT explaining your reasoning, then exactly ONE bash code \
block with the command to run, like this:

THOUGHT: The parser should skip blank lines; let me look at how it reads them.

```bash
grep -n "def parse" src/pkg/parser.py
```

Rules:
- Exactly one ```bash block per reply. Chain steps with && if you need several.
- Each command runs in a new shell in /work: `cd` and environment variables do not persist.
- No interactive programs (editors, pagers, `python` without arguments). Edit files with sed, \
`cat <<'EOF' > file`, or a short python script.
- Fix the library code, not the tests. Run the relevant tests to check your fix.
- When you are done, reply with a THOUGHT and this command alone: `echo {SUBMIT}`. You cannot \
continue after it."""
PROMPT = """Fix the issue below in the repository at /work.

<issue>
{issue}
</issue>"""
# Leaf's instruction: its system prompt is generic, so the facts of this sandbox go with the issue.
LEAF_PROMPT = """Fix the issue below in the repository at /work. Its dependencies are installed: `python` \
and `pytest` run in the project's environment. There is no network access. Fix the library code, not \
the tests.

<issue>
{issue}
</issue>"""
# With the issue in the system prompt, the user turn only starts the episode.
START = "Start with your first command."
LEAF_START = "Start working on the issue."
_BASH = re.compile(r"```(?:bash|sh)\s*\n(.*?)```", re.DOTALL)
# test files the grade restores before running the suite, so editing a test cannot pass it
_RESTORE_TESTS = (
    "git diff --name-only HEAD | grep -E '(^|/)(tests?/|testing/|test_[^/]*[.]py$|[^/]*_test[.]py$"
    "|conftest[.]py$)' | xargs -r git checkout -q HEAD --"
)
_start_lock: asyncio.Semaphore | None = None


class TinySweData(vf.TaskData):
    info: dict
    """The instance id, repo, suite time and the dataset paths of its patch and test lists."""


class TinySweTask(vf.Task[TinySweData]):
    pass


class TinySweConfig(vf.TasksetConfig):
    dataset_path: str
    """A staged task directory (stage_tasks.py): index.jsonl, tasks/, patches/."""
    index: str = "index.jsonl"
    """The task list in dataset_path; filter_tasks.py writes filtered ones next to index.jsonl."""
    num_examples: int | None = None
    """First N tasks after shuffling (all by default)."""
    shuffle_seed: int | None = 0
    """Shuffle the task order (index.jsonl is grouped by repo); None keeps it."""
    issue_in_system: bool = True
    """Put the issue in the system prompt, the only message KV-Streams eviction never drops: in the
    first user message it is evicted with the oldest turns (after 8 turns under r8), leaving the
    model without its task. False keeps it in the first user message."""
    harness: Literal["mini-swe", "leaf"] = "mini-swe"
    """The episode protocol and its prompts: mini-swe (one ```bash block per reply, with
    env.player.harness.id = "inline-chat") or leaf (Leaf's five function-calling tools, with
    env.player.harness.id = "leaf-harness")."""


class TinySweEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})
    image: str = "localhost/tinyswe-store:py312"
    podman: str = Field(default_factory=lambda: os.environ.get("TINYSWE_PODMAN", "podman"))
    """The podman command line, global flags included (on Mila: --root/--runroot on $SLURM_TMPDIR,
    --cgroup-manager=cgroupfs, --events-backend=file, overlay.ignore_chown_errors)."""
    socket_dir: str = Field(default_factory=lambda: os.environ.get("TINYSWE_SOCKETS", "/tmp/tinyswe-sockets"))
    """Host directory for the sandboxes' unix sockets (one subdirectory each)."""
    max_episode_steps: int = Field(50, ge=1)
    """Model turns (commands) before the episode is cut off and graded as it stands."""
    command_timeout: float = 60
    output_chars: int = 6000
    """Observation size: longer command output keeps its head and tail."""
    memory_mb: int = 2048
    """RLIMIT_DATA for every command and the grade."""
    max_parallel_starts: int = 8
    """Container starts in flight per env worker process (podman serializes on its locks)."""
    episode_log: str | None = None
    """Append one JSON line per finished episode (timings, reward, steps) to this file."""


class Sandbox:
    """One container of the shared image with the task's bug committed at /work."""

    def __init__(self, config: TinySweEnvConfig, info: dict):
        self.config, self.info = config, info
        self.name = f"tinyswe-{uuid.uuid4().hex[:12]}"
        self.sock_dir = Path(config.socket_dir) / self.name
        self.reader = self.writer = None

    async def start(self) -> float:
        global _start_lock
        _start_lock = _start_lock or asyncio.Semaphore(self.config.max_parallel_starts)
        t0 = time.time()
        self.sock_dir.mkdir(parents=True)
        repo, data = self.info["repo"], self.info["dataset_path"]
        commit = "git -c user.name=tinyswe -c user.email=tinyswe@localhost commit -qm base"
        if self.info.get("image"):
            # A SWE-smith image: the repo at /testbed (its conda env installs it in place, so it stays
            # there; /work links to it) and one git branch per task, i.e. every task's bug and fix.
            # Rebuild it as one commit holding this task's bug.
            image, vpy = self.info["image"], "/opt/miniconda3/envs/testbed/bin/python"
            path = "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/bin:/usr/bin:/bin"
            boot = (
                f"ln -s /testbed /work && cd /testbed && git apply /in/patches/{self.info['id']}.diff "
                f"&& rm -rf .git && git init -q && git add -A && {commit} "
                f"&& for py in /opt/miniconda3/bin/python3 {vpy} python3; do "
                f"command -v $py > /dev/null && exec $py /opt/cmdserver.py /sock/s; done"
            )
        else:
            # The bug must not be a commit of its own (`git log -p` would show the fix): rebuild the repo
            # as one commit that already holds it, keeping the release tag git-derived versions read.
            image, vpy, path = (
                self.config.image,
                f"/envs/{repo}/bin/python",
                f"/envs/{repo}/bin:/usr/local/bin:/usr/bin:/bin",
            )
            boot = (
                f"cp -a /repos/{repo} /work && cd /work && tag=$(git describe --tags --abbrev=0 2>/dev/null; true) "
                f"&& git apply /in/patches/{self.info['id']}.diff && rm -rf .git && git init -q && git add -A "
                f'&& {commit} && {{ [ -z "$tag" ] || git tag "$tag"; }} && exec /usr/local/bin/python /opt/cmdserver.py /sock/s'
            )
        argv = [
            *shlex.split(self.config.podman),
            "run",
            "-d",
            "--rm",
            "--network",
            "none",
            "--name",
            self.name,
            "--tmpfs",
            "/tmp:size=512m",
            "-e",
            f"VPY={vpy}",
            "-e",
            f"MEM_MB={self.config.memory_mb}",
            "-e",
            f"PATH={path}",
            "-v",
            f"{data}:/in:ro",
            "-v",
            f"{Path(__file__).parent / 'cmdserver.py'}:/opt/cmdserver.py:ro",
            "-v",
            f"{self.sock_dir}:/sock",
            image,
            "bash",
            "-c",
            boot,
        ]
        async with _start_lock:
            p = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
            )
            _, err = await p.communicate()
        if p.returncode:
            raise RuntimeError(f"podman run failed: {err.decode()[-300:]}")
        while not (self.sock_dir / "s").exists():
            if time.time() - t0 > 180:
                raise RuntimeError("sandbox did not come up in 180 s")
            await asyncio.sleep(0.05)
        self.reader, self.writer = await asyncio.open_unix_connection(str(self.sock_dir / "s"), limit=2**22)
        return time.time() - t0

    async def call(self, req: dict) -> dict:
        self.writer.write((json.dumps(req) + "\n").encode())
        await self.writer.drain()
        return json.loads(await self.reader.readline())

    async def close(self):
        if self.writer:
            self.writer.close()
        p = await asyncio.create_subprocess_exec(
            *shlex.split(self.config.podman),
            "rm",
            "-f",
            "-t",
            "0",
            self.name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await p.wait()
        try:
            for f in self.sock_dir.iterdir():
                f.unlink()
            self.sock_dir.rmdir()
        except OSError:
            pass


def _observation(r: dict) -> str:
    if "error" in r:
        return f"<error>{r['error']}</error>"
    note = "\n(the command timed out and was killed)" if r["rc"] == 124 else ""
    return f"<returncode>{r['rc']}</returncode>\n<output>\n{r['output']}</output>{note}"


class TinySweEnv(vf.Env[TinySweEnvConfig]):
    async def run(self, task, agents):
        info = task.data.info
        cfg = self.config
        started = time.time()
        sandbox = Sandbox(cfg, info)
        leaf = info.get("harness") == "leaf"
        harness_id = (cfg.player.harness.id if cfg.player.harness else None) or "null"
        if leaf != (harness_id == "leaf-harness"):
            raise ValueError(
                f"taskset.harness = {info.get('harness')!r} needs env.player.harness.id = "
                f"{'leaf-harness' if leaf else 'inline-chat'!r} (got {harness_id!r})"
            )
        steps = malformed = timeouts = 0
        submitted, exec_s, outcome, tools = False, 0.0, {}, Counter()
        try:
            boot_s = await sandbox.start()
            async with agents.player.interaction(task) as interaction:
                segment = await interaction.turn()
                while leaf and not segment.terminated:
                    steps += 1
                    calls = [c for m in segment.messages for c in (getattr(m, "tool_calls", None) or [])]
                    if not calls:  # Leaf's final answer
                        submitted = True
                        break
                    results = []
                    for c in calls:
                        try:
                            args = json.loads(c.arguments) if c.arguments else {}
                        except json.JSONDecodeError:
                            args = {}  # as in Leaf: the tool then reports its missing argument
                        if not isinstance(args, dict):
                            args = {}
                        r = await sandbox.call(
                            {
                                "op": "leaf",
                                "tool": c.name,
                                "args": args,
                                "timeout": cfg.command_timeout,
                                "max_timeout": 600,
                            }
                        )
                        exec_s += r.get("wall", 0)
                        timeouts += r.get("rc") == 124
                        tools[c.name] += 1
                        results.append(
                            {
                                "role": "tool",
                                "tool_call_id": c.id,
                                "name": c.name,
                                "content": r.get("output", f"<error>{r.get('error')}</error>"),
                            }
                        )
                    if steps >= cfg.max_episode_steps:
                        break
                    segment = await interaction.turn(results)
                while not leaf and not segment.terminated:
                    steps += 1
                    blocks = _BASH.findall(segment.last_reply)
                    if len(blocks) != 1:
                        malformed += 1
                        observation = (
                            f"Your reply had {len(blocks)} bash blocks. Reply with a THOUGHT and exactly "
                            f"ONE ```bash block (or `echo {SUBMIT}` when you are done)."
                        )
                    elif SUBMIT in blocks[0]:
                        submitted = True
                        break
                    else:
                        r = await sandbox.call(
                            {
                                "op": "sh",
                                "cmd": blocks[0],
                                "timeout": cfg.command_timeout,
                                "max_chars": cfg.output_chars,
                            }
                        )
                        exec_s += r.get("wall", 0)
                        timeouts += r.get("rc") == 124
                        observation = _observation(r)
                    if steps >= cfg.max_episode_steps:
                        break
                    segment = await interaction.turn(observation)
            trace = interaction.trace
            await sandbox.call({"op": "sh", "cmd": _RESTORE_TESTS, "timeout": 30})
            g = await sandbox.call(
                {"op": "grade", "task": f"/in/tasks/{info['id']}.json", "timeout": max(120, 5 * info["suite_s"])}
            )
            solved = g.get("rc") != 124 and g.get("f2p_failing") == 0 and g.get("n_p2p_failed") == 0
            outcome = {
                "boot_s": round(boot_s, 2),
                "grade_s": g.get("wall"),
                "f2p_failing": g.get("f2p_failing"),
                "n_f2p": g.get("n_f2p"),
                "n_p2p_failed": g.get("n_p2p_failed"),
                "grade_error": g.get("error"),
            }
        finally:
            await sandbox.close()
        reward = 1.0 if solved else 0.0
        trace.record_reward("task_success", reward, 1.0)
        outcome |= {
            "solved": solved,
            "submitted": submitted,
            "steps": steps,
            "malformed_replies": malformed,
            "command_timeouts": timeouts,
            "exec_s": round(exec_s, 2),
        }
        if leaf:
            outcome |= {"tool_calls": sum(tools.values()), **{f"calls_{k}": v for k, v in sorted(tools.items())}}
        trace.info["tinyswe"] = outcome
        if cfg.episode_log:
            line = {
                "start": started,
                "end": time.time(),
                "reward": reward,
                "id": info["id"],
                "repo": info["repo"],
                **outcome,
            }
            with open(cfg.episode_log, "a") as f:
                f.write(json.dumps(line) + "\n")


class TinySweTaskset(vf.Taskset[TinySweTask, TinySweConfig]):
    def load(self) -> Iterator[TinySweTask]:
        root = Path(self.config.dataset_path)
        rows = [json.loads(line) for line in open(root / self.config.index)]
        if not rows:
            raise ValueError(f"no tasks in {root / self.config.index}")
        if self.config.shuffle_seed is not None:
            random.Random(self.config.shuffle_seed).shuffle(rows)
        leaf = self.config.harness == "leaf"
        system, instruction, start = (
            (LEAF_SYSTEM_PROMPT, LEAF_PROMPT, LEAF_START) if leaf else (SYSTEM_PROMPT, PROMPT, START)
        )
        for row in rows[: self.config.num_examples]:
            issue = instruction.format(issue=row["problem_statement"])
            yield TinySweTask(
                TinySweData(
                    name=row["id"],
                    prompt=start if self.config.issue_in_system else issue,
                    system_prompt=system + "\n\n" + issue if self.config.issue_in_system else system,
                    info={
                        "id": row["id"],
                        "repo": row["repo"],
                        "suite_s": row["suite_s"],
                        "dataset_path": str(root.resolve()),
                        "harness": self.config.harness,
                        "image": row.get("image"),
                    },
                ),
                self.config.task,
            )
