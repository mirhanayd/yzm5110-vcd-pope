#!/usr/bin/env python
"""Offline control (no GPU): plain greedy decoding plus a constant logit bias toward "Yes".

Every run stores the clean first-step logits of "Yes" and "No". For greedy runs the answer is
argmax over the vocabulary; on POPE it is always one of the two (checked below: 100% agreement).
Adding a bias b to logit(Yes) - logit(No) shifts answers one way, toward "yes". For each split we
pick b so that the yes-ratio matches a VCD run, then compare F1. This is a diagnostic in the spirit
of Yin et al. (2025): it uses the yes-ratio of the VCD run on the test split (not the labels), so it
is not a deployable method, but it shows how much of the VCD gain a one-way shift alone explains.

    python scripts/yes_shift_control.py      # writes results/tables/yes_shift.md|csv
"""
import csv
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(ROOT, "results", "runs")
OUT = os.path.join(ROOT, "results", "tables")


def load(name):
    with open(os.path.join(RUNS, name, "answers.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def said_yes(r):
    a = r["text"].lower()
    return ("yes" in a) if r["label"] == "yes" else ("no" not in a)


def metrics(pred, labels):
    tp = sum(p and l == "yes" for p, l in zip(pred, labels))
    fp = sum(p and l == "no" for p, l in zip(pred, labels))
    fn = sum((not p) and l == "yes" for p, l in zip(pred, labels))
    tn = len(labels) - tp - fp - fn
    P, R = tp / (tp + fp), tp / (tp + fn)
    return dict(accuracy=100 * (tp + tn) / len(labels), precision=100 * P, recall=100 * R,
                f1=100 * 2 * P * R / (P + R), yes=100 * (tp + fp) / len(labels))


def main():
    rows = []
    for sp in ("random", "popular", "adversarial"):
        g = load(f"coco_{sp}_greedy_s1")
        labels = [r["label"] for r in g]
        margin = [r["first"]["clean_logit_yes"] - r["first"]["clean_logit_no"] for r in g]
        agree = sum((m > 0) == said_yes(r) for m, r in zip(margin, g)) / len(g)
        assert agree > 0.99, f"logit rule does not reproduce greedy answers on {sp}: {agree:.3f}"
        targets = [("VCD T=999, sampling, seed 1", f"coco_{sp}_vcd_a1_T999_s1"),
                   ("VCD T=999, greedy", f"coco_{sp}_vcd_greedy_a1_T999_s1"),
                   ("VCD T=500, sampling, seed 1", f"coco_{sp}_vcd_s1")]
        for name, run in targets:
            if not os.path.exists(os.path.join(RUNS, run)):
                continue
            v = load(run)
            byq = {r["question_id"]: r for r in v}
            t = metrics([said_yes(byq[r["question_id"]]) for r in g], labels)
            best = min((abs(metrics([m + b / 100 > 0 for m in margin], labels)["yes"] - t["yes"]), b / 100)
                       for b in range(0, 1201))
            b = best[1]
            s = metrics([m + b > 0 for m in margin], labels)
            rows.append([sp, name, round(t["yes"], 2), round(t["f1"], 2), b, round(s["yes"], 2), round(s["f1"], 2),
                         round(s["f1"] - t["f1"], 2), round(s["accuracy"] - t["accuracy"], 2)])
    header = ["split", "matched VCD run", "VCD yes %", "VCD F1", "bias b", "greedy+b yes %", "greedy+b F1",
              "F1 difference (greedy+b minus VCD)", "accuracy difference"]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "yes_shift.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(os.path.join(OUT, "yes_shift.md"), "w") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    print(open(os.path.join(OUT, "yes_shift.md")).read())


if __name__ == "__main__":
    main()
