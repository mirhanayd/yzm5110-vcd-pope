#!/usr/bin/env python
"""List the questions VCD (T=500) gets wrong in all 3 seeds while regular decoding gets them right
in all 3 seeds, and the reverse. Writes results/tables/failure_examples.md (no GPU)."""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(ROOT, "results", "runs")


def load(n):
    with open(os.path.join(RUNS, n, "answers.jsonl")) as f:
        return {r["question_id"]: r for r in map(json.loads, filter(str.strip, f))}


def ok(r):
    a = r["text"].lower()
    return ("yes" in a) if r["label"] == "yes" else ("no" in a)


lines = ["# Consistent changes between regular decoding and VCD (T=500), 3 seeds each", "",
         "Broken = regular right in all 3 seeds, VCD wrong in all 3. Fixed = the reverse.", "",
         "| split | change | image | question | label | regular (seed 1) | VCD (seed 1) |", "|---|---|---|---|---|---|---|"]
for sp in ("random", "popular", "adversarial"):
    R = [load(f"coco_{sp}_regular_s{s}") for s in (1, 2, 3)]
    V = [load(f"coco_{sp}_vcd_s{s}") for s in (1, 2, 3)]
    for change, cond in (("broken", lambda q: all(ok(r[q]) for r in R) and not any(ok(v[q]) for v in V)),
                         ("fixed", lambda q: not any(ok(r[q]) for r in R) and all(ok(v[q]) for v in V))):
        for q in sorted(R[0]):
            if cond(q):
                r = R[0][q]
                lines.append(f"| {sp} | {change} | {r['image']} | {r['prompt']} | {r['label']} | {r['text']} | {V[0][q]['text']} |")
out = os.path.join(ROOT, "results", "tables", "failure_examples.md")
open(out, "w").write("\n".join(lines) + "\n")
print(f"wrote {out}: {len(lines) - 6} rows")
