"""In-container command server: the sandbox's main process. Serves one client on a unix socket,
one JSON request per line, so the agent's commands never go through `podman exec`.

  {"op": "sh", "cmd": "...", "timeout": s, "max_chars": n} -> {"rc", "wall", "output"}
  {"op": "grade", "task": "/in/tasks/<id>.json", "timeout": s}
      -> {"rc", "wall", "f2p_failing", "n_f2p", "n_p2p_failed", "p2p_failed"}
  {"op": "leaf", "tool": "Read|Write|Edit|Glob|Bash", "args": {...}, "timeout": s} -> {"rc", "wall", "output"}
      one tool call of the Leaf harness (leaf_harness), in Leaf's own output formats

Runs with the image's own Python; commands run in /work with the repo's venv first on PATH, under
RLIMIT_DATA (MEM_MB) and a wall timeout. RLIMIT_AS alone at 2-4 GB breaks threaded test suites
(glibc reserves address space per thread arena), so the address-space cap is only a 16 GB backstop.
"""

import glob
import json
import os
import resource
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

VPY = os.environ["VPY"]
MEM = int(os.environ.get("MEM_MB", 2048)) * 2**20
PYTEST = [VPY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly", "--continue-on-collection-errors"]


def limits():
    os.setsid()  # its own process group, so a timeout kills everything it started
    resource.setrlimit(resource.RLIMIT_DATA, (MEM, MEM))
    resource.setrlimit(resource.RLIMIT_AS, (16 * 2**30, 16 * 2**30))


def run(argv, timeout):
    t = time.time()
    p = subprocess.Popen(argv, cwd="/work", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=limits)
    try:
        out, _ = p.communicate(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        out, _ = p.communicate()
        rc = 124
    return {"rc": rc, "wall": round(time.time() - t, 3)}, out.decode(errors="replace")


def clip(text, n):
    if len(text) <= n:
        return text
    return f"{text[: n // 2]}\n... [{len(text) - n} characters elided] ...\n{text[-n // 2 :]}"


# Leaf tools, ported from FrogNano's stdlib tool runner (github.com/microsoft/FrogNano,
# frognano/harness/leaf/tool_runner.py, MIT): same arguments, output formats and truncation, so a
# model trained in Leaf sees what it saw there. Differences: paths resolve against /work; Bash runs
# `bash -c` (a login shell's profile would reset PATH off the repo's venv) under the sandbox's limits.
LEAF_READ_CHARS, LEAF_BASH_CHARS, LEAF_GLOB_LIMIT = 20000, 30000, 1000


def leaf_truncate(value, limit):
    return value if len(value) <= limit else f"{value[:limit]}\n[truncated {len(value) - limit} chars]"


def leaf_path(args, key="file_path"):
    value = args.get(key) or args.get("path")
    if not value:
        raise ValueError(f"{key} is required")
    path = os.path.expanduser(str(value))
    return path if os.path.isabs(path) else os.path.join("/work", path)


def leaf_display(path):
    rel = os.path.relpath(os.path.realpath(path), "/work")
    return path if rel.startswith("..") else rel


def leaf_int(value, default):
    return default if value in (None, "") else int(value)


def leaf_tool(name, args, timeout):
    """One Leaf tool call -> (rc, output): rc is the Bash command's exit code (124 on a timeout),
    0 for the file tools; a tool's error is its output, as in Leaf."""
    try:
        if name == "Read":
            path = leaf_path(args)
            lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
            offset = max(1, leaf_int(args.get("offset"), 1))
            limit = leaf_int(args.get("limit"), 0)
            end = len(lines) if limit <= 0 else min(len(lines), offset - 1 + limit)
            body = "\n".join(f"{n}. {line}" for n, line in enumerate(lines[offset - 1 : end], start=offset))
            return 0, leaf_truncate(f"{leaf_display(path)}\n{body}", LEAF_READ_CHARS)
        if name == "Write":
            path, content = leaf_path(args), str(args.get("content", ""))
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            open(path, "w", encoding="utf-8").write(content)
            return 0, f"Wrote {leaf_display(path)} ({len(content)} bytes)."
        if name == "Edit":
            path = leaf_path(args)
            old, new = str(args.get("old_string") or ""), str(args.get("new_string") or "")
            if not old:
                raise ValueError("old_string must be non-empty")
            text = open(path, encoding="utf-8", errors="replace").read()
            count = text.count(old)
            replace_all = args.get("replace_all") is True or str(args.get("replace_all") or "").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if count == 0:
                raise ValueError("old_string was not found")
            if count != 1 and not replace_all:
                raise ValueError(
                    f"old_string matched {count} times; set replace_all=true or use a more specific old_string"
                )
            open(path, "w", encoding="utf-8").write(
                text.replace(old, new) if replace_all else text.replace(old, new, 1)
            )
            changed = count if replace_all else 1
            return 0, f"Edited {leaf_display(path)} ({changed} replacement{'s' if changed != 1 else ''})."
        if name == "Glob":
            pattern = str(args.get("pattern") or "")
            if not pattern:
                raise ValueError("pattern is required")
            root = leaf_path({"path": args.get("path") or "."}, "path")
            matches = sorted(
                leaf_display(m) for m in glob.glob(os.path.join(root, pattern), recursive=True) if os.path.isfile(m)
            )
            more = f"\n... {len(matches) - LEAF_GLOB_LIMIT} more files" if len(matches) > LEAF_GLOB_LIMIT else ""
            return 0, "\n".join(matches[:LEAF_GLOB_LIMIT]) + more if matches else "No files matched."
        if name == "Bash":
            command = str(args.get("command") or "")
            if not command:
                raise ValueError("command is required")
            t = time.time()
            p = subprocess.Popen(
                ["bash", "-c", command], cwd="/work", stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=limits
            )
            try:
                out, err = p.communicate(timeout=timeout)
                head, rc = f"Exit code: {p.returncode}", p.returncode
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, 9)
                out, err = p.communicate()
                head, rc = f"Timed out after {time.time() - t:.1f}s", 124
            text = f"{head}\nSTDOUT:\n{out.decode(errors='replace')}\nSTDERR:\n{err.decode(errors='replace')}"
            return rc, leaf_truncate(text, LEAF_BASH_CHARS)
        return 0, f"Unknown tool: {name}"
    except Exception as e:  # Leaf reports a tool's error to the model as its output
        return 0, f"{type(e).__name__}: {e}"


def leaf_timeout(args, default):
    """Leaf's Bash timeout argument: seconds, or milliseconds when over 1000 (or as timeout_ms)."""
    if args.get("timeout_ms") is not None:
        return max(0.1, float(args["timeout_ms"]) / 1000)
    raw = args.get("timeout")
    if raw is None:
        return default
    value = float(raw)
    return max(0.1, value / 1000 if value > 1000 else value)


def junit_pass(path):
    ok = set()
    if os.path.exists(path):
        for tc in ET.parse(path).getroot().iter("testcase"):
            if not any(c.tag in ("failure", "error", "skipped") for c in tc):
                ok.add(f"{tc.get('classname')}::{tc.get('name')}")
    return ok


def handle(req):
    if req["op"] == "sh":
        r, out = run(["bash", "-c", req["cmd"]], req["timeout"])
        return r | {"output": clip(out, req.get("max_chars", 10000))}
    if req["op"] == "grade":
        task = json.load(open(req["task"]))
        xd = ["-n0"] if subprocess.run([VPY, "-c", "import xdist"], capture_output=True).returncode == 0 else []
        if os.path.exists("/tmp/grade.xml"):
            os.unlink("/tmp/grade.xml")
        r, _ = run(PYTEST + xd + ["--junitxml=/tmp/grade.xml"], req["timeout"])
        ok = junit_pass("/tmp/grade.xml")
        p2p_failed = [t for t in task["PASS_TO_PASS"] if t not in ok]
        return r | {
            "f2p_failing": sum(t not in ok for t in task["FAIL_TO_PASS"]),
            "n_f2p": len(task["FAIL_TO_PASS"]),
            "n_p2p_failed": len(p2p_failed),
            "p2p_failed": p2p_failed[:20],
        }
    if req["op"] == "leaf":
        t = time.time()
        args = req.get("args") or {}
        rc, out = leaf_tool(req["tool"], args, min(leaf_timeout(args, req["timeout"]), req.get("max_timeout", 600)))
        return {"rc": rc, "wall": round(time.time() - t, 3), "output": out}
    return {"error": f"unknown op {req['op']}"}


def main():
    path = sys.argv[1]
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path + ".tmp")
    os.rename(path + ".tmp", path)  # the socket appears only once it is bound
    srv.listen(1)
    conn, _ = srv.accept()
    f = conn.makefile("rw")
    for line in f:
        try:
            out = handle(json.loads(line))
        except Exception as e:  # report, keep serving
            out = {"error": repr(e)[:300]}
        f.write(json.dumps(out) + "\n")
        f.flush()


if __name__ == "__main__":
    main()
