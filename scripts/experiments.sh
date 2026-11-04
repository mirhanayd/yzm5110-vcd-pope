#!/usr/bin/env bash
# Exact commands for every run reported in the report. Every run resumes if interrupted.
#   bash scripts/experiments.sh smoke                    # 100 questions x 3 configs
#   bash scripts/experiments.sh phase1                   # reproduction + controls (COCO) and gate tuning (dev)
#   TAU=.. SKIP=.. bash scripts/experiments.sh phase2    # improvement on COCO + clean timing runs
#   bash scripts/experiments.sh tables
# WORKERS (default 4) = models sharing the GPU in parallel (A100 80GB: 4; L4/T4: 1).
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS="${WORKERS:-4}"
SPLITS="random popular adversarial"
SEEDS="1 2 3"     # VCD reports 5 runs; 3 seeds here (compute budget), stated in the report
DEV="--dataset aokvqa --split popular --exclude-images-from data/pope/coco/coco_pope_random.json"
TAUS="0.6 0.8 0.9 0.95"
mkdir -p results/jobs

case "${1:-}" in
smoke)
  {
    echo "--dataset coco --split random --mode regular --limit 100"
    echo "--dataset coco --split random --mode vcd --limit 100"
    echo "--dataset coco --split random --mode vcd --gate lossless --limit 100"
  } > results/jobs/smoke.txt
  python scripts/run_many.py results/jobs/smoke.txt --workers 1
  ;;
phase1)
  {
    # dev runs first, so the gate threshold is known even if the session ends early
    for s in 1 2; do
      echo "$DEV --mode vcd --seed $s"
      for t in $TAUS; do echo "$DEV --mode vcd --gate margin --tau $t --seed $s"; done
    done
    # reproduction of VCD Table 1 (LLaVA-1.5, MSCOCO) with controls
    for sp in $SPLITS; do
      for s in $SEEDS; do
        echo "--dataset coco --split $sp --mode regular --seed $s"   # paper's baseline: direct sampling
        echo "--dataset coco --split $sp --mode vcd --seed $s"       # alpha=1, beta=0.1, T=500
        echo "--dataset coco --split $sp --mode apc --seed $s"       # control: APC alone (alpha=0)
      done
      echo "--dataset coco --split $sp --mode greedy --seed 1"       # control: greedy decoding
      echo "--dataset coco --split $sp --mode vcd_greedy --seed 1"
    done
  } > results/jobs/phase1.txt
  python scripts/run_many.py results/jobs/phase1.txt --workers "$WORKERS"
  python scripts/pick_tau.py
  ;;
phase2)
  : "${TAU:?set TAU from pick_tau.py}"; : "${SKIP:?set SKIP from pick_tau.py}"
  {
    for sp in $SPLITS; do
      for s in $SEEDS; do
        echo "--dataset coco --split $sp --mode vcd --gate lossless --seed $s"              # exact; must match VCD
        echo "--dataset coco --split $sp --mode vcd --gate margin --tau $TAU --seed $s"     # lossy, tau from dev
        echo "--dataset coco --split $sp --mode vcd --gate random --skip-prob $SKIP --seed $s"  # control
      done
    done
  } > results/jobs/phase2.txt
  python scripts/run_many.py results/jobs/phase2.txt --workers "$WORKERS"
  # latency/memory: one model alone on the GPU, first 1000 questions of COCO-random
  T="--dataset coco --split random --seed 1 --limit 1000 --out-dir results/timing"
  {
    echo "$T --mode regular"
    echo "$T --mode vcd"
    echo "$T --mode vcd --gate lossless"
    echo "$T --mode vcd --gate margin --tau $TAU"
    echo "$T --mode vcd --gate random --skip-prob $SKIP"
  } > results/jobs/timing.txt
  python scripts/run_many.py results/jobs/timing.txt --workers 1
  ;;
tables)
  python scripts/make_tables.py
  python scripts/make_tables.py --runs-dir results/timing --prefix timing_
  ;;
*) echo "usage: $0 smoke|phase1|phase2|tables"; exit 1;;
esac
