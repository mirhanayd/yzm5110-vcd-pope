#!/usr/bin/env bash
# Exact commands for every run reported in the report. Each command resumes if interrupted.
# Usage: bash scripts/experiments.sh <phase>     phases: smoke | repro | dev | test | tables
set -euo pipefail
cd "$(dirname "$0")/.."
R="python scripts/run_pope.py"
SPLITS="random popular adversarial"
SEEDS="${SEEDS:-1 2 3}"          # VCD reports 5 runs; we use 3 (compute budget) - see report
DEV="--dataset aokvqa --split popular --exclude-images-from data/pope/coco/coco_pope_random.json"

case "${1:-}" in
smoke)   # ~5 min: checks the pipeline and measures speed on this GPU
  $R --dataset coco --split random --mode regular --limit 100
  $R --dataset coco --split random --mode vcd --limit 100
  $R --dataset coco --split random --mode vcd --gate lossless --limit 100
  ;;
repro)   # Phase 1 - reproduction of VCD Table 1 (LLaVA-1.5, MSCOCO) + controls
  for sp in $SPLITS; do
    for s in $SEEDS; do
      $R --dataset coco --split $sp --mode regular --seed $s   # "Regular" = direct sampling (paper's baseline)
      $R --dataset coco --split $sp --mode vcd --seed $s       # VCD, alpha=1, beta=0.1, T=500
      $R --dataset coco --split $sp --mode apc --seed $s       # control: APC alone (alpha=0)
    done
    $R --dataset coco --split $sp --mode greedy --seed 1       # control: greedy decoding (deterministic)
    $R --dataset coco --split $sp --mode vcd_greedy --seed 1
  done
  ;;
dev)     # Phase 2a - tune the margin gate on A-OKVQA POPE (dev), never on COCO (test)
  for s in 1 2; do
    $R $DEV --mode vcd --seed $s
    for tau in 0.5 0.7 0.8 0.9 0.95 0.99; do $R $DEV --mode vcd --gate margin --tau $tau --seed $s; done
  done
  python scripts/pick_tau.py
  ;;
test)    # Phase 2b - improvement vs reproduced baseline on COCO; set TAU and SKIP from pick_tau.py
  : "${TAU:?set TAU from pick_tau.py}"; : "${SKIP:?set SKIP from pick_tau.py}"
  for sp in $SPLITS; do
    for s in $SEEDS; do
      $R --dataset coco --split $sp --mode vcd --gate lossless --seed $s          # exact, should match VCD 100%
      $R --dataset coco --split $sp --mode vcd --gate margin --tau $TAU --seed $s  # lossy, tuned on dev
      $R --dataset coco --split $sp --mode vcd --gate random --skip-prob $SKIP --seed $s  # control: same skip rate, random
    done
  done
  ;;
tables)
  python scripts/make_tables.py
  ;;
*) echo "usage: $0 smoke|repro|dev|test|tables"; exit 1;;
esac
