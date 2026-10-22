"""A queryable workspace for an agent: markdown files it writes and a ledger the
env writes for it, both searchable with one tool.

The model calls the tools as tags in its reply, next to its action; they cost no
game step and their results arrive at the top of the next observation (so a
turn is still one model call, and KV-Streams' turn window is unchanged). Every
observation shows only a compact index of the workspace, so what the agent has
stored does not grow the per-turn context: it asks for what it needs.

Game-agnostic: an env appends ledger lines (``log``) for what happened; the
agent searches, reads and writes.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_WRITE = re.compile(r'<write\s+file\s*=\s*"([^"]*)"\s*>(.*?)</write>', re.DOTALL)
_APPEND = re.compile(r'<append\s+file\s*=\s*"([^"]*)"\s*>(.*?)</append>', re.DOTALL)
_READ = re.compile(r"<read>\s*(.*?)\s*</read>", re.DOTALL)
_SEARCH = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_PATH = re.compile(r"^[a-z0-9][a-z0-9_\-/]{0,40}\.md$")
_COORDS = re.compile(r"\((-?\d+),\s*(-?\d+)\)")
_NEAR = re.compile(r"\bnear:\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?")
LEDGER = "ledger"


def instructions(files: bool, ledger: bool, file_chars: int) -> str:
    """The system-prompt section that explains the workspace tools."""
    lines = [
        "Workspace tools (optional; write them in the same reply as your action, they",
        "cost no game step; their results appear at the top of your next observation):",
        "- <search>water</search> finds lines in your workspace that contain every",
        '  word; "near:30,28" keeps lines about places within 8 steps of (30, 28);',
        '  "re:" starts a regular expression, e.g. <search>re:(iron|diamond)</search>.',
    ]
    if files:
        lines += [
            "- <read>map.md</read> shows a file"
            + ("; <read>ledger</read> shows the latest ledger entries." if ledger else "."),
            '- <write file="plan.md">text</write> creates or replaces a file;',
            f'  <append file="map.md">text</append> adds to one (files hold at most {file_chars} characters).',
        ]
    elif ledger:
        lines.append("- <read>ledger</read> shows the latest ledger entries.")
    if ledger:
        lines += [
            "The ledger is written for you every step: what you saw and where (water, stone,",
            "ores, lava, trees, cows, tables), what you collected, made and placed, and when",
            "and how you died.",
        ]
    lines.append(
        "The workspace index is shown at the top of every observation. Older turns may drop out "
        "of your context; your workspace does not."
    )
    return "\n".join(lines)


def _clean(text: str) -> str:
    return "\n".join(" ".join(line.split()) for line in text.strip().splitlines() if line.strip())


@dataclass
class Workspace:
    files_enabled: bool = True
    ledger_enabled: bool = True
    max_files: int = 20
    file_chars: int = 2000
    result_lines: int = 10
    read_lines: int = 30
    tools_per_reply: int = 4
    files: dict[str, str] = field(default_factory=dict)
    ledger: list[str] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    searches: list[dict] = field(default_factory=list)

    def log(self, line: str) -> None:
        if self.ledger_enabled:
            self.ledger.append(line)

    # ---- tools ---------------------------------------------------------------------

    def apply(self, reply: str, step: int) -> list[str]:
        """Run the workspace tools in `reply`; returns the lines to show next turn."""
        out: list[str] = []
        if self.files_enabled:
            for path, text in _WRITE.findall(reply)[: self.tools_per_reply]:
                out.append(self._write(path, text, append=False))
            for path, text in _APPEND.findall(reply)[: self.tools_per_reply]:
                out.append(self._write(path, text, append=True))
        for target in _READ.findall(reply)[: self.tools_per_reply]:
            out += self._read(target.strip().lower())
        for query in _SEARCH.findall(reply)[: self.tools_per_reply]:
            out += self._search(query.strip(), step)
        return out

    def _write(self, path: str, text: str, append: bool) -> str:
        path = path.strip().lower()
        verb = "append" if append else "write"
        self.counts[verb] += 1
        if not _PATH.match(path) or path == LEDGER:
            self.counts["bad_path"] += 1
            return f'{verb} "{path}": not a valid file name (use names like plan.md or notes/map.md).'
        text = _clean(text)
        if path not in self.files and len(self.files) >= self.max_files:
            return f'{verb} "{path}": workspace full ({self.max_files} files); replace or append to one.'
        body = f"{self.files[path]}\n{text}".strip() if append and path in self.files else text
        cut = len(body) > self.file_chars
        self.files[path] = body[-self.file_chars :] if append else body[: self.file_chars]
        lines = self.files[path].count("\n") + 1
        return f"{'Appended to' if append else 'Wrote'} {path} ({lines} lines" + (", cut to fit)." if cut else ").")

    def _read(self, target: str) -> list[str]:
        self.counts["read"] += 1
        if target == LEDGER and self.ledger_enabled:
            entries = self.ledger[-self.read_lines :]
            head = f"ledger (last {len(entries)} of {len(self.ledger)} entries):"
            return [head, *(f"  {e}" for e in entries)] if entries else ["ledger: empty."]
        if target not in self.files:
            return [f"read {target}: no such file." + (f" Files: {', '.join(sorted(self.files))}." if self.files else "")]
        body = self.files[target].splitlines()
        shown = body[: self.read_lines]
        return [f"{target}:", *(f"  {line}" for line in shown)] + (
            [f"  ... ({len(body) - len(shown)} more lines)"] if len(body) > len(shown) else []
        )

    def _search(self, query: str, step: int) -> list[str]:
        self.counts["search"] += 1
        near = None
        if m := _NEAR.search(query):
            near = (int(m.group(1)), int(m.group(2)))
            query = (query[: m.start()] + query[m.end() :]).strip()
        regex = None
        if query.startswith("re:"):
            try:
                regex = re.compile(query[3:].strip()[:100], re.IGNORECASE)
            except re.error as e:
                self.searches.append({"step": step, "query": query, "hits": 0, "error": True})
                return [f"search {query!r}: bad regular expression ({e})."]
        terms = [] if regex else query.lower().split()

        def matches(line: str) -> bool:
            if regex is not None:
                if not regex.search(line):
                    return False
            elif not all(t in line.lower() for t in terms):
                return False
            if near is not None:
                return any(
                    max(abs(int(x) - near[0]), abs(int(y) - near[1])) <= 8 for x, y in _COORDS.findall(line)
                )
            return True

        hits = [f"ledger: {line}" for line in reversed(self.ledger) if matches(line)]  # newest first
        for path in sorted(self.files):
            hits += [f"{path}:{i}: {line}" for i, line in enumerate(self.files[path].splitlines(), 1) if matches(line)]
        label = " ".join(part for part in (query, f"near:{near[0]},{near[1]}" if near else "") if part)
        self.searches.append({"step": step, "query": label, "hits": len(hits), "top": hits[0] if hits else None})
        if not hits:
            return [f"search {label!r}: no matches."]
        shown = hits[: self.result_lines]
        more = f" (showing {len(shown)})" if len(hits) > len(shown) else ""
        return [f"search {label!r}: {len(hits)} matches{more}", *(f"  {h}" for h in shown)]

    # ---- views ---------------------------------------------------------------------

    def index(self) -> str:
        parts = []
        if self.files_enabled:
            listed = ", ".join(f"{p} ({self.files[p].count(chr(10)) + 1} lines)" for p in sorted(self.files))
            parts.append(f"files: {listed or 'none'}")
        if self.ledger_enabled:
            parts.append(f"ledger: {len(self.ledger)} entries")
        return "Workspace: " + "; ".join(parts)

    def summary(self) -> dict:
        return {
            "files": {p: t[:600] for p, t in sorted(self.files.items())},
            "ledger_entries": len(self.ledger),
            "ledger_tail": self.ledger[-10:],
            "counts": dict(self.counts),
            "searches": self.searches[:200],
        }
