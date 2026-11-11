#!/usr/bin/env python
"""Aggregate results/runs/*/ into the tables used in the report.

    python scripts/make_tables.py                     # writes results/tables/*.md and *.csv

* main.{md,csv}: one row per (dataset, split, configuration); mean +- std over seeds of
  POPE accuracy / precision / recall / F1 / yes-ratio (official scorer), plus a 95%
  bootstrap CI over questions for F1 (pooled over seeds), mean latency, noisy forward
  passes and noisy prefills per question, skip rate and peak GPU memory.
* agreement.{md,csv}: per question agreement between each gated run and the ungated VCD
  run with the same seed (a lossless gate should agree on ~100% of questions).
* paired.{md,csv}: VCD-vs-baseline paired comparison per split: number of questions fixed
  (wrong -> right) and broken (right -> wrong), split by label, and an exact McNemar p-value.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import random
import re
import statistics as stats
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(ROOT, "results", "runs")
OUT = os.path.join(ROOT, "results", "tables")
PREFIX = ""


def correct(rec):
    a = rec["text"].lower().strip()
    return ("yes" in a) if rec["label"] == "yes" else ("no" in a)


def said_yes(rec):
    a = rec["text"].lower().strip()
    return ("yes" in a) if rec["label"] == "yes" else ("no" not in a)


def f1_of(recs):
    tp = sum(r["label"] == "yes" and said_yes(r) for r in recs)
    fp = sum(r["label"] == "no" and said_yes(r) for r in recs)
    fn = sum(r["label"] == "yes" and not said_yes(r) for r in recs)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return 2 * p * r / (p + r) if p + r else 0


def bootstrap_ci(recs_by_seed, n_boot=1000, seed=0):
    """Resample questions (same resample for all seeds), F1 averaged over seeds."""
    rng = random.Random(seed)
    qids = sorted(recs_by_seed[0].keys())
    vals = []
    for _ in range(n_boot):
        idx = [rng.choice(qids) for _ in qids]
        vals.append(stats.mean(f1_of([rs[i] for i in idx]) for rs in recs_by_seed))
    vals.sort()
    return 100 * vals[int(0.025 * n_boot)], 100 * vals[int(0.975 * n_boot) - 1]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load_runs():
    runs = []
    for meta_path in sorted(glob.glob(os.path.join(RUNS, "*", "meta.json"))):
        d = os.path.dirname(meta_path)
        meta = json.load(open(meta_path))
        recs = {}
        for line in open(os.path.join(d, "answers.jsonl")):
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                recs[r["question_id"]] = r
        if len(recs) < meta["n_questions"]:
            print(f"skip incomplete run {os.path.basename(d)} ({len(recs)}/{meta['n_questions']})")
            continue
        name = os.path.basename(d)
        config = re.sub(r"_s\d+(?=(_n\d+)?$)", "", name)  # strip seed, keep the _n<limit> suffix
        runs.append(dict(name=name, config=config, meta=meta, recs=recs, seed=meta["args"]["seed"]))
    return runs


def fmt(xs):
    return f"{stats.mean(xs):.2f}" + (f" ± {stats.stdev(xs):.2f}" if len(xs) > 1 else "")


def write(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    name = PREFIX + name
    with open(os.path.join(OUT, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(os.path.join(OUT, name + ".md"), "w") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    print(f"wrote {OUT}/{name}.md ({len(rows)} rows)")


def main():
    global RUNS, PREFIX
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=RUNS)
    ap.add_argument("--prefix", default="", help="prefix for output table names, e.g. timing_")
    args = ap.parse_args()
    RUNS, PREFIX = args.runs_dir, args.prefix
    runs = load_runs()
    groups = defaultdict(list)
    for r in runs:
        groups[r["config"]].append(r)

    header = ["config", "seeds", "accuracy", "precision", "recall", "F1", "F1 95% CI", "yes %",
              "latency ms/q (mean)", "latency ms/q (median)", "GPU shared", "noisy fwd/q", "noisy prefill/q", "skip rate %", "peak GB"]
    rows = []
    for cfg, rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["seed"])
        m = [r["meta"]["metrics_official"] for r in rs]
        lo, hi = bootstrap_ci([r["recs"] for r in rs])
        rows.append([
            cfg, len(rs), fmt([x["accuracy"] for x in m]), fmt([x["precision"] for x in m]),
            fmt([x["recall"] for x in m]), fmt([x["f1"] for x in m]), f"[{lo:.2f}, {hi:.2f}]",
            fmt([x["yes_ratio"] for x in m]),
            fmt([1000 * r["meta"]["mean_latency_s"] for r in rs]),
            fmt([1000 * r["meta"].get("median_latency_s", r["meta"]["mean_latency_s"]) for r in rs]),
            "yes" if any(r["meta"].get("concurrent_jobs", 1) > 1 for r in rs) else "no",
            f"{stats.mean(r['meta']['noisy_forwards_per_q'] for r in rs):.3f}",
            f"{stats.mean(r['meta']['noisy_prefills_per_q'] for r in rs):.3f}",
            f"{100 * stats.mean(r['meta']['skip_rate_steps'] for r in rs):.1f}",
            max((r["meta"]["peak_memory_gb"] or 0) for r in rs),
        ])
    write("main", header, rows)

    # agreement of gated runs with ungated VCD (same dataset/split/seed/sampling mode)
    by_key = {(r["config"], r["seed"]): r for r in runs}
    arows = []
    for r in runs:
        mm = re.match(r"(.+?)_(vcd|vcd_greedy)_(lossless|margin[\d.]+|random[\d.]+)(.*)$", r["config"])
        if not mm:
            continue
        base = by_key.get((f"{mm.group(1)}_{mm.group(2)}{mm.group(4)}", r["seed"]))
        if not base:
            continue
        same = sum(r["recs"][q]["text"] == base["recs"][q]["text"] for q in r["recs"])
        arows.append([r["name"], base["name"], f"{100 * same / len(r['recs']):.2f}"])
    write("agreement", ["gated run", "ungated VCD run", "identical answers %"], arows)

    # paired comparisons against baselines with the same seed
    prow = []
    pairs = [("vcd", "regular"), ("vcd_greedy", "greedy"), ("apc", "regular"), ("vcd", "apc")]
    for r in runs:
        for new, old in pairs:
            # the optional suffix covers runs at a non-default alpha / noise step (e.g. _a1_T999)
            mm = re.match(rf"(.+)_{new}(?:_a[\d.]+_T\d+)?$", r["config"])
            if not mm:
                continue
            base = by_key.get((f"{mm.group(1)}_{old}", r["seed"])) or by_key.get((f"{mm.group(1)}_{old}", 1))
            if not base:
                continue
            fixed = defaultdict(int)
            broken = defaultdict(int)
            for q, rec in r["recs"].items():
                c_new, c_old = correct(rec), correct(base["recs"][q])
                if c_new and not c_old:
                    fixed[rec["label"]] += 1
                if c_old and not c_new:
                    broken[rec["label"]] += 1
            b, c = sum(fixed.values()), sum(broken.values())
            prow.append([r["name"], base["name"], fixed["yes"], fixed["no"], broken["yes"], broken["no"],
                         f"{mcnemar_exact(b, c):.2g}"])
    write("paired", ["run", "baseline", "fixed (label yes)", "fixed (label no)", "broken (label yes)",
                     "broken (label no)", "McNemar p"], prow)


if __name__ == "__main__":
    main()
