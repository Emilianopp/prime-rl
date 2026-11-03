"""Bundle extract_traces.py outputs into one Trace Inspector page.

usage: build_viewer.py <out.html> <label>=<data.json> [<label>=<data.json> ...]
Each run becomes a tab; the viewer picks per-env rendering (tinyswe, crafter, or generic).
"""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
out, specs = sys.argv[1], sys.argv[2:]
runs = []
for spec in specs:
    label, path = spec.split("=", 1)
    d = json.load(open(path))
    d["label"] = label
    runs.append(d)
blob = json.dumps(runs, separators=(",", ":")).replace("</", "<\\/")  # keep </script> out of the JSON
page = open(os.path.join(here, "viewer.html")).read().replace("__DATA__", blob)
open(out, "w").write(page)
print(f"{out}: {len(runs)} runs, {len(page) / 1e6:.1f} MB")
