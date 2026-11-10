#!/usr/bin/env python
"""Run a list of run_pope.py configurations with N worker processes sharing one GPU.

Each worker loads the model ONCE and runs its share of the jobs one after another, which
avoids reloading 14 GB of weights for every run. With --workers > 1 several 7B models share
the GPU (each ~15 GB), which raises throughput on large GPUs (A100 80GB) but makes per-question
latency unrepresentative; such runs are tagged concurrent_jobs > 1 in meta.json and latency is
reported only from runs with --workers 1 (the "timing" phase).

    python scripts/run_many.py jobs.txt --workers 4        # one job per line = run_pope.py arguments
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def worker(jobs, wid, nworkers):
    sys.path.insert(0, HERE)
    import run_pope  # noqa: E402

    lm = None
    mine = [j for i, j in enumerate(jobs) if i % nworkers == wid]
    for k, line in enumerate(mine):
        a = run_pope.parse_args(shlex.split(line))
        print(f"[worker {wid}] job {k + 1}/{len(mine)}: {line}", flush=True)
        lm = run_pope.run(a, lm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_file")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    jobs = [l.strip() for l in open(a.jobs_file) if l.strip() and not l.startswith("#")]
    if a.worker_id is not None:
        worker(jobs, a.worker_id, a.workers)
        return
    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    # Each fp16 LLaVA-1.5-7B worker needs ~15 GB; cap the worker count to what the GPU can hold
    # (asked via nvidia-smi so this parent process does not create a CUDA context of its own).
    try:
        total_gb = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                               text=True).split()[0]) / 1024
        fit = max(1, int(total_gb // 17))
        if a.workers > fit:
            print(f"GPU has {total_gb:.0f} GB: running {fit} worker(s) instead of {a.workers}", flush=True)
            a.workers = fit
    except Exception:
        pass
    env = dict(os.environ, VCDPOPE_CONCURRENT=str(a.workers))
    tag = os.path.splitext(os.path.basename(a.jobs_file))[0]
    procs = []
    for w in range(a.workers):
        log = open(os.path.join(ROOT, "results", "logs", f"{tag}_worker{w}.log"), "a")
        procs.append(subprocess.Popen([sys.executable, __file__, a.jobs_file, "--workers", str(a.workers),
                                       "--worker-id", str(w)], stdout=log, stderr=subprocess.STDOUT, env=env))
    print(f"{len(jobs)} jobs, {a.workers} workers; logs in results/logs/{tag}_worker*.log", flush=True)
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(60)
        print(f"  {int((time.time() - t0) / 60)} min elapsed, {sum(p.poll() is None for p in procs)} workers running",
              flush=True)
    codes = [p.returncode for p in procs]
    print("done, exit codes:", codes, f"({(time.time() - t0) / 60:.1f} min)")
    if any(codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
