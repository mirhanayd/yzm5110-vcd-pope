#!/usr/bin/env python
"""Run LLaVA-1.5 on one POPE split with a given decoding mode and save per-question outputs.

Example:
  python scripts/run_pope.py --dataset coco --split random --mode vcd --seed 1
  python scripts/run_pope.py --dataset coco --split random --mode vcd --gate lossless --seed 1
  python scripts/run_pope.py --dataset coco --split random --mode regular --seed 1 --limit 50   # smoke test

Outputs go to results/runs/<run_name>/ : answers.jsonl (one line per question, with
per-sample decoding statistics and latency), meta.json (config, versions, git revision,
GPU, peak memory, wall time). Re-running the same command resumes an interrupted run.
Randomness is seeded per question (seed, question_id), so runs are resumable and
different decoding modes see the same noise image v' for the same question.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from vcdpope.decoding import GATES, MODES, DecodeConfig, decode  # noqa: E402
from vcdpope.model import build_prompt, load_llava, preprocess_image, tokenize_with_image  # noqa: E402
from vcdpope.noise import add_diffusion_noise  # noqa: E402
from vcdpope.pope import load_questions, score_official, score_strict  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def git_rev():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def run_name(a):
    name = f"{a.dataset}_{a.split}_{a.mode}"
    if a.gate != "none":
        name += f"_{a.gate}"
        if a.gate == "margin":
            name += f"{a.tau:g}"
        if a.gate == "random":
            name += f"{a.skip_prob:g}"
    if a.mode in ("vcd", "vcd_greedy") and (a.alpha != 1.0 or a.noise_step != 500):
        name += f"_a{a.alpha:g}_T{a.noise_step}"
    if a.mode in ("vcd", "vcd_greedy", "apc", "apc_greedy") and a.beta != 0.1:
        name += f"_b{a.beta:g}"
    if a.load_in_4bit:
        name += "_4bit"
    return name + f"_s{a.seed}" + (f"_n{a.limit}" if a.limit else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="coco", choices=["coco", "aokvqa", "gqa"])
    p.add_argument("--split", default="random", choices=["random", "popular", "adversarial"])
    p.add_argument("--question-file", default=None)
    p.add_argument("--image-folder", default=None)
    p.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--mode", default="regular", choices=MODES)
    p.add_argument("--gate", default="none", choices=GATES)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--skip-prob", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--noise-step", type=int, default=500)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke tests)")
    p.add_argument("--exclude-images-from", default=None,
                   help="POPE file whose images are removed from this split (keeps dev and test images disjoint)")
    p.add_argument("--out-dir", default=os.path.join(ROOT, "results", "runs"))
    p.add_argument("--name", default=None)
    a = p.parse_args()

    qfile = a.question_file or os.path.join(ROOT, "data", "pope", a.dataset, f"{a.dataset}_pope_{a.split}.json")
    ifolder = a.image_folder or os.path.join(ROOT, "data", "images", "gqa" if a.dataset == "gqa" else "coco_val2014")
    questions = load_questions(qfile)
    if a.exclude_images_from:
        excl = {q["image"] for q in load_questions(a.exclude_images_from)}
        questions = [q for q in questions if q["image"] not in excl]
    if a.limit:
        questions = questions[: a.limit]

    out = os.path.join(a.out_dir, a.name or run_name(a))
    os.makedirs(out, exist_ok=True)
    ans_path = os.path.join(out, "answers.jsonl")
    done = {}
    if os.path.exists(ans_path):
        with open(ans_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["question_id"]] = r
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"[run] {out}: {len(done)} done, {len(todo)} to go")

    cfg = DecodeConfig(mode=a.mode, alpha=a.alpha, beta=a.beta, max_new_tokens=a.max_new_tokens, gate=a.gate,
                       tau=a.tau, skip_prob=a.skip_prob)
    lm = load_llava(a.model_id, a.dtype, a.load_in_4bit)
    tok = lm.tokenizer
    cfg.eos_token_id = tok.eos_token_id
    yes_no = (tok("Yes").input_ids[-1], tok("No").input_ids[-1])  # "▁Yes", "▁No" as emitted after "ASSISTANT:"
    cuda = lm.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    t_run = time.perf_counter()
    with open(ans_path, "a") as fout:
        for i, q in enumerate(todo):
            qid = q["question_id"]
            torch.manual_seed(a.seed * 1_000_003 + qid)  # seeds CPU (noise) and CUDA (sampling) RNGs
            gate_rng = random.Random(a.seed * 7_919 + qid)
            input_ids = tokenize_with_image(build_prompt(q["text"]), lm).to(lm.device)
            pix = preprocess_image(os.path.join(ifolder, q["image"]), lm)
            pix_cd = add_diffusion_noise(pix, a.noise_step)  # always drawn, so RNG streams match across modes
            pix = pix.to(lm.device, lm.dtype)
            pix_cd = pix_cd.to(lm.device, lm.dtype) if cfg.contrastive else None
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            gen, st = decode(lm.model, input_ids, pix, pix_cd, cfg, gate_rng, yes_no)
            if cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            text = tok.decode(gen, skip_special_tokens=True).strip()
            rec = dict(question_id=qid, image=q["image"], prompt=q["text"], label=q["label"], text=text,
                       latency_s=round(dt, 5), steps=st.steps, clean_forwards=st.clean_forwards,
                       noisy_forwards=st.noisy_forwards, noisy_prefills=st.noisy_prefills,
                       skipped_steps=st.skipped_steps, lossless_steps=st.lossless_steps, first=st.first)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            done[qid] = rec
            if (i + 1) % 100 == 0:
                print(f"[run] {i + 1}/{len(todo)}  last={text!r}  {dt:.3f}s")
    wall = time.perf_counter() - t_run

    answers = {qid: r["text"] for qid, r in done.items()}
    lat = [r["latency_s"] for r in done.values()]
    meta = dict(
        args=vars(a), question_file=qfile, n_questions=len(questions), git_revision=git_rev(),
        model_revision=getattr(lm.model.config, "_commit_hash", None),
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      transformers=__import__("transformers").__version__),
        gpu=torch.cuda.get_device_name(0) if cuda else "cpu",
        peak_memory_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3) if cuda else None,
        wall_time_s_this_session=round(wall, 1), n_new_this_session=len(todo),
        mean_latency_s=round(sum(lat) / len(lat), 5) if lat else None,
        noisy_forwards_per_q=round(sum(r["noisy_forwards"] for r in done.values()) / len(done), 4),
        noisy_prefills_per_q=round(sum(r["noisy_prefills"] for r in done.values()) / len(done), 4),
        skip_rate_steps=round(sum(r["skipped_steps"] for r in done.values()) / max(1, sum(r["steps"] for r in done.values())), 4),
        metrics_official=score_official(questions, answers),
        metrics_strict=score_strict(questions, answers),
        command=" ".join([os.path.basename(sys.executable)] + sys.argv),
    )
    meta_path = os.path.join(out, "meta.json")
    if os.path.exists(meta_path) and len(todo) < len(questions):
        meta["previous_sessions"] = json.load(open(meta_path)).get("previous_sessions", []) + [
            {k: v for k, v in json.load(open(meta_path)).items() if k in ("gpu", "peak_memory_gb", "wall_time_s_this_session")}]
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(json.dumps(meta["metrics_official"]), f"peak_mem={meta['peak_memory_gb']}GB", f"mean_lat={meta['mean_latency_s']}s")


if __name__ == "__main__":
    main()
