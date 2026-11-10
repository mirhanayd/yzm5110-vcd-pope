#!/usr/bin/env python
"""Draw the two report figures from the saved outputs (no GPU).

    python scripts/make_figures.py      # writes results/figures/fig1_f1_by_split.png and fig2_latency.png

Figure 1: POPE F1 per split for regular decoding, VCD at T=500 and T=999, the paper's VCD
(Leng et al., 2024, Table 1) and greedy decoding with a constant yes-shift (yes_shift_control.py).
Figure 2: mean latency per question from the timing runs (one model alone on the GPU).
"""
import glob
import json
import os
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import yes_shift_control as ysc  # noqa: E402

RUNS, TIMING = os.path.join(ROOT, "results", "runs"), os.path.join(ROOT, "results", "timing")
OUT = os.path.join(ROOT, "results", "figures")
SPLITS = ["random", "popular", "adversarial"]
PAPER_VCD = {"random": 87.16, "popular": 85.06, "adversarial": 81.33}  # Leng et al. (2024), Table 1
GREY, BLUE, BLACK = "#b9b9ae", "#2b7bd6", "#111111"


def f1(run):
    return json.load(open(os.path.join(RUNS, run, "meta.json")))["metrics_official"]["f1"]


def mean_f1(prefix):
    return st.mean(f1(os.path.basename(os.path.dirname(m))) for m in glob.glob(os.path.join(RUNS, prefix + "_s[0-9]", "meta.json")))


def shifted_greedy_f1(split):
    g = ysc.load(f"coco_{split}_greedy_s1")
    labels = [r["label"] for r in g]
    margin = [r["first"]["clean_logit_yes"] - r["first"]["clean_logit_no"] for r in g]
    v = {r["question_id"]: r for r in ysc.load(f"coco_{split}_vcd_a1_T999_s1")}
    target = ysc.metrics([ysc.said_yes(v[r["question_id"]]) for r in g], labels)["yes"]
    b = min(range(0, 1201), key=lambda k: abs(ysc.metrics([m + k / 100 > 0 for m in margin], labels)["yes"] - target)) / 100
    return ysc.metrics([m + b > 0 for m in margin], labels)["f1"]


def fig1():
    series = [("Regular (mine, 3 seeds)", lambda s: mean_f1(f"coco_{s}_regular"), dict(mfc="none", mec=GREY, ms=8, mew=2)),
              ("VCD, T 500 (mine, 3 seeds)", lambda s: mean_f1(f"coco_{s}_vcd"), dict(mfc=GREY, mec=GREY, ms=8)),
              ("VCD, T 999 (mine, seed 1)", lambda s: f1(f"coco_{s}_vcd_a1_T999_s1"), dict(mfc=BLUE, mec=BLUE, ms=9)),
              ("VCD (paper)", lambda s: PAPER_VCD[s], dict(mfc="none", mec=BLACK, ms=16, mew=1.5)),
              ("Greedy + yes-shift (mine)", shifted_greedy_f1, dict(mfc=BLACK, mec=BLACK, ms=8))]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for label, fn, style in series:
        xs = [fn(s) for s in SPLITS]
        ax.plot(xs, range(len(SPLITS)), "o", linestyle="none", label=label, **style)
    ax.set_yticks(range(len(SPLITS)), SPLITS)
    ax.invert_yaxis()
    ax.set_xlim(76, 90)
    ax.set_xlabel("F1")
    ax.grid(axis="x", color="#e5e5e5")
    ax.set_title("At the paper's noise level VCD matches its published F1,\nbut a yes-shift on greedy decoding scores higher", loc="left", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False, fontsize=8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_f1_by_split.png"), dpi=200)


def fig2():
    order = [("Regular decoding", "coco_random_regular_s1_n1000"), ("VCD", "coco_random_vcd_s1_n1000"),
             ("VCD + lossless gate", "coco_random_vcd_lossless_s1_n1000"),
             ("VCD + lossy gate (tau 0.6)", "coco_random_vcd_margin0.6_s1_n1000"),
             ("VCD + random gate", "coco_random_vcd_random0.9581_s1_n1000")]
    ms = [json.load(open(os.path.join(TIMING, run, "meta.json")))["mean_latency_s"] * 1000 for _, run in order]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    colors = [BLUE if "lossless" in n else GREY for n, _ in order]
    ax.barh(range(len(order)), ms, color=colors)
    for i, v in enumerate(ms):
        ax.text(v + 2, i, f"{v:.0f} ms", va="center", fontsize=9)
    ax.axvline(ms[0], color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(order)), [n for n, _ in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 215)
    ax.set_xlabel("mean latency per question (ms), one A100, 1,000 COCO-random questions")
    ax.set_title("Lossless gate: same answers as VCD at 115 ms instead of 191", loc="left", fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_latency.png"), dpi=200)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1()
    fig2()
    print("wrote", sorted(os.listdir(OUT)))
