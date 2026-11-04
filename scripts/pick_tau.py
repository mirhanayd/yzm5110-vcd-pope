#!/usr/bin/env python
"""Choose the margin-gate threshold tau on the DEV split only (A-OKVQA POPE, images disjoint
from COCO). Rule fixed before looking at test results: the smallest tau (= most skipping)
whose dev F1, averaged over the available seeds, is within --max-drop points of ungated VCD.
Prints tau* and the measured skip rate, which is then used for the random-gate control."""
import argparse, glob, json, os, re, statistics as st
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="aokvqa_popular_vcd")
ap.add_argument("--max-drop", type=float, default=0.5)
a = ap.parse_args()
f1, skip = defaultdict(list), defaultdict(list)
for m in glob.glob(os.path.join(ROOT, "results", "runs", a.prefix + "*", "meta.json")):
    name = os.path.basename(os.path.dirname(m))
    mm = re.match(rf"{a.prefix}(?:_margin([\d.]+))?_s\d+$", name)
    if not mm:
        continue
    tau = float(mm.group(1)) if mm.group(1) else None
    meta = json.load(open(m))
    f1[tau].append(meta["metrics_official"]["f1"])
    skip[tau].append(meta["skip_rate_steps"])
assert None in f1, "run ungated VCD on the dev split first"
ref = st.mean(f1[None])
print(f"ungated VCD dev F1 = {ref:.2f} ({len(f1[None])} seeds)")
ok = []
for tau in sorted(t for t in f1 if t is not None):
    d = st.mean(f1[tau]) - ref
    print(f"tau={tau:<5} F1={st.mean(f1[tau]):.2f} (diff {d:+.2f})  skip rate={100 * st.mean(skip[tau]):.1f}%")
    if d >= -a.max_drop:
        ok.append(tau)
if ok:
    t = min(ok)
    print(f"\nTAU={t}  SKIP_PROB={st.mean(skip[t]):.4f}   <- use these for the test runs")
else:
    print("\nno tau meets the criterion; report this and use lossless only")
