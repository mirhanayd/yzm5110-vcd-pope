"""POPE data loading and metrics.

``score_official`` reproduces experiments/eval/eval_pope.py of the VCD repo (substring
matching: a "yes" label counts as correct if "yes" occurs in the answer, a "no" label if
"no" occurs). ``score_strict`` parses the first word only and reports unparseable answers
separately; it is used as a sanity check of the official scorer.
"""
from __future__ import annotations

import json
import re


def load_questions(path: str, limit: int | None = None) -> list[dict]:
    with open(path) as f:
        qs = [json.loads(line) for line in f if line.strip()]
    return qs[:limit] if limit else qs


def _metrics(tp, tn, fp, fn, n, yes, extra=None):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    m = dict(accuracy=(tp + tn) / n, precision=prec, recall=rec, f1=f1, yes_ratio=yes / n, n=n)
    if extra:
        m.update(extra)
    return {k: (round(v * 100, 2) if isinstance(v, float) else v) for k, v in m.items()}


def score_official(gt: list[dict], answers: dict[int, str]) -> dict:
    tp = tn = fp = fn = yes = 0
    for q in gt:
        a = answers[q["question_id"]].lower().strip()
        if q["label"] == "yes":
            if "yes" in a:
                tp += 1; yes += 1
            else:
                fn += 1
        else:
            if "no" in a:
                tn += 1
            else:
                fp += 1; yes += 1
    return _metrics(tp, tn, fp, fn, len(gt), yes)


def parse_strict(text: str) -> str | None:
    m = re.match(r"\W*(yes|no)\b", text.lower())
    return m.group(1) if m else None


def score_strict(gt: list[dict], answers: dict[int, str]) -> dict:
    tp = tn = fp = fn = yes = unk = 0
    for q in gt:
        p = parse_strict(answers[q["question_id"]])
        if p is None:
            unk += 1
            p = "no"  # unparseable answers count as "no", reported separately
        else:
            yes += p == "yes"
        if q["label"] == "yes":
            tp += p == "yes"; fn += p != "yes"
        else:
            tn += p == "no"; fp += p != "no"
    return _metrics(tp, tn, fp, fn, len(gt), yes, {"unparsed": unk / len(gt)})
