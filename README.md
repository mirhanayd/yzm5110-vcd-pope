# Visual Contrastive Decoding on POPE: reproduction and confidence-gated VCD

YZM5110 Multimodal Learning research project.

**Paper reproduced:** S. Leng et al., *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding*, CVPR 2024 ([arXiv:2311.16922](https://arxiv.org/abs/2311.16922), [official code](https://github.com/DAMO-NLP-SG/VCD)).

## Question

VCD reduces object hallucination in LLaVA-1.5 by contrasting the next-token logits for the real image `v` with those for a noise-distorted image `v'`:

```
p_vcd(y) = softmax( (1+α)·logit(y|v,x) − α·logit(y|v',x) )   restricted to   { y : p(y|v,x) ≥ β · max_w p(w|v,x) }
```

1. **Reproduction.** Does VCD improve POPE accuracy/F1 over regular decoding for LLaVA-1.5-7B on MSCOCO (paper: Table 1)? We also check how much of the gain is explained by two simpler effects reported by later work ([Yin et al., NeurIPS 2025](https://arxiv.org/abs/2504.10020)): the adaptive plausibility constraint (APC) alone, and greedy decoding. We also track the "yes" ratio.
2. **Improvement: cost.** VCD needs a second forward pass with the noisy image (including a second ~600-token prefill) for every generated token. When the APC keeps only one candidate token, the output is the clean argmax whatever the noisy logits are. In that case the noisy pass is redundant and can be skipped **without changing the output**. We measure how often this happens. We also measure the accuracy/cost trade-off of a lossy version that skips when the clean top-token probability is ≥ τ (τ tuned on a separate dev split). A random gate at the same skip rate serves as a control.

## Results in brief

| | Random F1 | Popular F1 | Adversarial F1 |
|---|---|---|---|
| Paper, VCD (Table 1) | 87.16 | 85.06 | 81.33 |
| Ours, regular decoding (3 seeds) | 81.16 | 79.85 | 77.43 |
| Ours, VCD T=500 (official script value, 3 seeds) | 83.26 | 81.78 | 79.16 |
| Ours, VCD T=999 (paper value, 1 seed) | 87.12 | 85.13 | 81.43 |
| Ours, greedy + yes-shift matched to VCD's yes-ratio (offline diagnostic) | 89.03 | 86.94 | 83.12 |
| Ours, greedy + yes-shift tuned on dev labels (b = 0.01) | 85.55 | 84.42 | 82.17 |

* VCD reproduces within 0.12 F1 at the paper's noise level T=999; at T=500 the gain is about a third.
* A constant logit shift toward "Yes" on greedy decoding, matched to VCD's yes-ratio, beats VCD on every split; tuned on held-out dev labels instead, the shift is about 0 and loses on 2 of 3 splits (`scripts/yes_shift_control.py`).
* The lossless gate gives identical answers to VCD on all 9,000 test questions and cuts latency from 191 ms to 115 ms per question (regular decoding: 95 ms) on an A100.
* The lossy gate (tau=0.6, tuned on A-OKVQA dev) does no better than a random gate at the same skip rate.

Full tables: `results/tables/`. Report: `Aydinli_YZM5110_Research.pdf`.

## Repository layout

```
src/vcdpope/noise.py      diffusion noise for v' (adapted from official VCD code, Apache-2.0)
src/vcdpope/model.py      LLaVA-1.5 (HF port) loading, prompt and tokenization identical to the official eval script
src/vcdpope/decoding.py   token-by-token decoding: regular, greedy, VCD, APC-only, gated VCD
src/vcdpope/pope.py       POPE metrics (official substring scorer + strict first-word scorer)
scripts/run_pope.py       one run = one (split, decoding mode, seed); resumable; writes results/runs/<name>/
scripts/experiments.sh    the exact commands of every reported run (phases: smoke, phase1, phase2, tables)
scripts/run_many.py       runs a job list with N workers sharing one GPU, loading the model once per worker
scripts/pick_tau.py       chooses the gate threshold on the dev split only
scripts/make_tables.py    builds results/tables/*.md|csv from the saved runs
scripts/yes_shift_control.py  offline yes-shift control from saved first-step logits (no GPU)
scripts/failure_examples.py   questions VCD consistently breaks or fixes (no GPU)
scripts/make_figures.py       Figures 1 and 2 of the report (no GPU)
scripts/check_tokenizer.py  verifies our prompt token ids == original LLaVA tokenizer ids
tests/                    CPU tests with a tiny random LLaVA (exactness of the lossless gate, KV-cache catch-up)
notebooks/colab_run.ipynb Colab runner (GPU)
results/runs, results/tables  saved outputs used in the report
```

## Setup

```bash
git clone <this repo> && cd <repo>
pip install -r requirements.txt          # torch: use the build matching your CUDA
python scripts/download_data.py          # POPE files (pinned VCD commit) + ~1000 COCO val2014 images
python -m pytest -q tests                # ~1 min on CPU
```

GPU: LLaVA-1.5-7B in fp16 needs ~15 GB (L4 / A100 / RTX 3090+). `--load-in-4bit` runs on 16 GB T4s but changes the model (reported separately).
On Colab, open `notebooks/colab_run.ipynb`.

## Reproducing the reported numbers

The reported results come from exactly these commands (GPU: one A100 80 GB on Colab Pro; `notebooks/colab_run.ipynb` runs them):

```bash
TAU=0.6 SKIP=0.9581 WORKERS=4 bash scripts/experiments.sh all   # 53 accuracy runs + 5 timing runs, ~4.8 A100 hours
WORKERS=4 bash scripts/experiments.sh t999                      # follow-up: VCD with the paper's noise step T=999 (4 runs)
bash scripts/experiments.sh tables                              # CPU only: every table and figure from the saved outputs
```

`TAU` and `SKIP` were chosen on the dev split (A-OKVQA POPE, images disjoint from COCO) by `scripts/pick_tau.py` in an earlier session whose run files were lost; its console output is in `results/notes/`. The `all` phase re-runs the dev split with seed 1 so the choice can be checked. The older `phase1` / `phase2` phases split the same jobs over two sessions.

**Without a large GPU:** `python -m pytest -q tests` (CPU, ~1 min) checks the lossless gate on a tiny random LLaVA, and `bash scripts/experiments.sh smoke` (100 questions x 3 decoders, a few minutes on any 16 GB GPU) checks the full pipeline. Everything else needs the 7B model in fp16 (~15 GB GPU memory) and about 5 A100 hours.

**Tested versions:** Python 3.13.16, transformers 4.51.3, torch 2.11.0+cu130 (main runs) and 2.14.1+cu130 (T=999 runs), NVIDIA A100-SXM4-80GB. Each run's `meta.json` records the exact versions, git revision and model revision (`llava-hf/llava-1.5-7b-hf` @ `b234b80`). Commit hashes recorded before a message-only history rewrite are mapped in `results/notes/commit_hash_map.txt`.

**Outputs used in the report:** `results/runs/` (accuracy), `results/timing/` (latency, memory), `results/tables/` (all tables, incl. `yes_shift.md` and `failure_examples.md`), `results/figures/` (Figures 1 and 2), `results/logs_archive/` (worker logs and job lists).

Accuracy runs use 4 models in parallel on one A100-80GB (`WORKERS=4`); their latencies are not used. Latency and memory are reported only from `results/timing/`, where one model runs alone on the GPU after untimed warm-up.

Every run directory contains `answers.jsonl` (answer, latency, forward-pass counts, first-step logits per question) and `meta.json` (all arguments, git revision, HF model revision, library versions, GPU, peak memory, wall time, metrics).

## Known differences from the original setup

| Aspect | Paper / official code | Here | Effect |
|---|---|---|---|
| Model implementation | `liuhaotian/llava-v1.5-7b` + LLaVA codebase, transformers 4.31 | HF port `llava-hf/llava-1.5-7b-hf`, transformers 4.51 | same weights; small numerical differences possible |
| Prompt / tokenization | `llava_v1` template, `tokenizer_image_token` | same template; token ids verified identical (`scripts/check_tokenizer.py`) | none expected |
| Image preprocessing | CLIP resize 336 + center crop, noise added after normalization | same | none expected |
| Decoding loop | monkey-patched HF `sample()` | own loop, same formula and APC (log-space cutoff, as in code) | equal in exact arithmetic |
| β | paper: 0.1; `llava1.5_pope.bash`: 0.2 | 0.1 | follows the paper |
| noise step T | paper appendix: 999 for POPE; official POPE script: 500 | 500 in the main runs, 999 in a follow-up (`experiments.sh t999`) | explains the gap to the paper (see Results) |
| runs | 5 runs, mean ± std | 3 seeds (compute budget) | wider uncertainty |
| seeding | one global seed per run | seed per (run seed, question id) | resumable runs, paired comparisons across modes |
| max new tokens | 1024 | 32 (answers are one word, then EOS) | none expected on POPE |
| vocabulary | 32000 | HF port pads the LM head to 32064; we use the first 32000 logits | none |

## Licenses and attribution

Our code is MIT-licensed (`LICENSE`). `src/vcdpope/noise.py` is adapted from the official VCD repository (Apache-2.0, `third_party/VCD_LICENSE`). POPE question files (`data/pope/`) are copied from the VCD repository at commit `d6568ff` (Apache-2.0); they are the POPE splits of Li et al., EMNLP 2023. COCO images are subject to the [COCO terms of use](https://cocodataset.org/#termsofuse) and are not redistributed here. LLaVA-1.5 weights are subject to their own license (Llama 2 community license).
