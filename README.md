# Visual Contrastive Decoding on POPE: reproduction and gated VCD

Research project for YZM5110 Multimodal Learning.

Paper: S. Leng et al., *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding*, CVPR 2024. Links: [arXiv:2311.16922](https://arxiv.org/abs/2311.16922) and the [official code](https://github.com/DAMO-NLP-SG/VCD).

## Question

VCD targets object hallucination in LLaVA-1.5. It compares the next-token logits for the real image `v` with those for a noisy copy `v'`:

```
p_vcd(y) = softmax( (1+α)·logit(y|v,x) − α·logit(y|v',x) )   restricted to   { y : p(y|v,x) ≥ β · max_w p(w|v,x) }
```

This repository asks two things.

**1. Does the gain reproduce?** The paper reports that VCD beats regular decoding on POPE (Table 1, LLaVA-1.5-7B on MSCOCO). I rerun that comparison. Two simpler effects could explain part of the gain ([Yin et al., 2025](https://arxiv.org/abs/2504.10020)): the adaptive plausibility constraint (APC) on its own, and greedy decoding. Each gets a control run, and I also track the "yes" ratio.

**2. Can VCD be cheaper?** VCD runs a second forward pass on the noisy image for every token, including a second prefill of about 600 tokens. When the APC keeps a single candidate token, the output is the clean argmax and the noisy logits cannot change it, so skipping the noisy pass there is lossless. I measure how often that happens. A lossy variant skips whenever the clean top-token probability reaches τ, with τ tuned on a separate dev split; its control is a random gate at the same skip rate.

## Results in brief

| | Random F1 | Popular F1 | Adversarial F1 |
|---|---|---|---|
| Paper, VCD (Table 1) | 87.16 | 85.06 | 81.33 |
| Mine, regular decoding (3 seeds) | 81.16 | 79.85 | 77.43 |
| Mine, VCD T=500 (official script value, 3 seeds) | 83.26 | 81.78 | 79.16 |
| Mine, VCD T=999 (paper value; 2 seeds, popular 1 seed) | 87.16 | 85.13 | 81.59 |
| Mine, greedy | 85.54 | 84.38 | 82.24 |
| Mine, VCD greedy T=999 | 88.14 | 85.91 | 82.13 |
| Mine, greedy + yes-shift matched to VCD's yes-ratio (offline diagnostic) | 89.03 | 86.94 | 83.12 |
| Mine, greedy + yes-shift tuned on dev labels (b = 0.01) | 85.55 | 84.42 | 82.17 |

* At the paper's noise level (T=999) VCD lands within 0.26 F1 of the published scores. At T=500 the gain is about a third of the published one. The official POPE script uses 500.
* VCD greedy at T=999 scores 88.14 F1 on the random split in my run. Table 7 of the paper reports 88.02 ± 0.30.
* Against plain greedy, the contrast at T=999 helps on random (exact McNemar p = 0.004). On popular I cannot tell the two apart in accuracy (p = 0.71). Adversarial accuracy drops by 2.0 points (p = 0.002). Most fixed answers have the label "yes", while the broken ones mostly have "no". The counts are in `results/tables/paired.md`.
* A constant logit shift toward "Yes" on greedy decoding beats VCD on every split when it is matched to VCD's yes-ratio. I matched it on test data, so read it as a diagnostic. Tuned on dev labels the shift is about 0 and loses on 2 of 3 splits. My script for this is `scripts/yes_shift_control.py`.
* The lossless gate returns the same answers as VCD on all 9,000 test questions. I measured 191 ms per question for VCD on an A100 and 115 ms with the gate. Regular decoding takes 95 ms.
* The lossy gate (τ = 0.6, tuned on A-OKVQA dev) does no better than a random gate at the same skip rate.

Full tables are in `results/tables/`. The report is `Aydinli_YZM5110_Research.pdf`. My class slides are in `Aydinli_YZM5110_Presentation.pdf`.

## Repository layout

```
Aydinli_YZM5110_Research.pdf      the report
Aydinli_YZM5110_Presentation.pdf  the 7 slides of the class presentation
src/vcdpope/noise.py      diffusion noise for v' (adapted from official VCD code, Apache-2.0)
src/vcdpope/model.py      LLaVA-1.5 (HF port) loading; prompt + tokenization identical to the official eval script
src/vcdpope/decoding.py   token-by-token decoding: regular, greedy, VCD, APC-only, gated VCD
src/vcdpope/pope.py       POPE metrics (official substring scorer + strict first-word scorer)
scripts/run_pope.py       one run = one (split, decoding mode, seed); resumable; writes results/runs/<name>/
scripts/experiments.sh    the exact commands of every reported run (phases: all, t999, t999b, smoke, tables)
scripts/run_many.py       runs a job list with N workers sharing one GPU, loading the model once per worker
scripts/pick_tau.py       chooses the gate threshold on the dev split only
scripts/make_tables.py    builds results/tables/*.md|csv from the saved runs
scripts/yes_shift_control.py  offline yes-shift control from saved first-step logits (no GPU)
scripts/failure_examples.py   questions VCD consistently breaks or fixes (no GPU)
scripts/make_figures.py       Figures 1 and 2 of the report (no GPU)
scripts/check_tokenizer.py  verifies the prompt token ids == original LLaVA tokenizer ids
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

LLaVA-1.5-7B in fp16 needs about 15 GB of GPU memory. An L4 is enough. `--load-in-4bit` fits a 16 GB T4 but changes the model, so I report it separately. On Colab, open `notebooks/colab_run.ipynb`.

## Reproducing the reported numbers

These commands produced the reported results. The notebook runs the same ones.

```bash
TAU=0.6 SKIP=0.9581 WORKERS=4 bash scripts/experiments.sh all   # 53 accuracy runs + 5 timing runs, ~4.8 A100 hours
WORKERS=4 bash scripts/experiments.sh t999                      # follow-up: VCD with the paper's noise step T=999 (4 runs)
bash scripts/experiments.sh t999b                               # second follow-up at T=999 (4 runs, 54 min with one worker on an L4)
bash scripts/experiments.sh tables                              # CPU only: every table and figure from the saved outputs
```

`TAU` and `SKIP` come from the dev split (A-OKVQA POPE, images disjoint from COCO). `scripts/pick_tau.py` chose them in an earlier session whose run files were lost; its console output is kept in `results/notes/`. The `all` phase reruns the dev split with seed 1, so the choice can be checked.

**No large GPU?** The CPU tests (`python -m pytest -q tests`) check the lossless gate on a tiny random LLaVA in about a minute. The smoke phase (`bash scripts/experiments.sh smoke`) checks the full pipeline on 100 questions with 3 decoders; it needs a 16 GB GPU and a few minutes. Everything else needs the 7B model in fp16 and about 5 A100 hours.

**Tested versions.** I used Python 3.13.16 and transformers 4.51.3 in every run. The main runs had torch 2.11.0+cu130 on an A100-SXM4-80GB. By the T=999 runs Colab had moved to torch 2.14.1+cu130; `t999` ran on an A100 and `t999b` on an L4. Each `meta.json` records the library versions next to the git revision and the model revision (`llava-hf/llava-1.5-7b-hf` @ `b234b80`). The `t999b` runs record the revision as `0a175eb-dirty`: the Colab working copy had uncommitted changes. Some hashes were recorded before a message-only history rewrite. `results/notes/commit_hash_map.txt` maps them to the current ones.

**Outputs used in the report.** Accuracy runs are in `results/runs/`, latency and memory runs in `results/timing/`. All tables, including `yes_shift.md` and `failure_examples.md`, are under `results/tables/`. Figures 1 and 2 are in `results/figures/`, and worker logs and job lists are archived in `results/logs_archive/`.

Accuracy runs on the A100 ran 4 models in parallel (`WORKERS=4`), so their latencies are not used. Latency and memory come only from `results/timing/`, where one model runs alone on the GPU after an untimed warm-up.

Each run directory has two files. `answers.jsonl` has one line per question (3,000 in total) with the answer, its latency, forward-pass counts and first-step logits. `meta.json` holds the arguments, git and model revisions, library versions, GPU, peak memory, wall time and metrics.

## Known differences from the original setup

| Aspect | Paper / official code | Here | Effect |
|---|---|---|---|
| Model implementation | `liuhaotian/llava-v1.5-7b` + LLaVA codebase, transformers 4.31 | HF port `llava-hf/llava-1.5-7b-hf`, transformers 4.51 | same weights; small numerical differences possible |
| Prompt / tokenization | `llava_v1` template, `tokenizer_image_token` | same template; token ids verified identical (`scripts/check_tokenizer.py`) | none expected |
| Image preprocessing | CLIP resize 336 + center crop, noise added after normalization | same | none expected |
| Decoding loop | monkey-patched HF `sample()` | own loop; same formula, same APC (log-space cutoff, as in code) | equal in exact arithmetic |
| β | paper: 0.1; `llava1.5_pope.bash`: 0.2 | 0.1 | follows the paper |
| noise step T | paper appendix: 999 for POPE; official POPE script: 500 | 500 in the main runs, 999 in two follow-ups (`experiments.sh t999`, `t999b`) | explains the gap to the paper (see Results) |
| runs | 5 runs, mean ± std | 3 seeds (compute budget) | wider uncertainty |
| seeding | one global seed per run | seed per (run seed, question id) | resumable runs, paired comparisons across modes |
| max new tokens | 1024 | 32 (answers are one word, then EOS) | none expected on POPE |
| vocabulary | 32000 | HF port pads the LM head to 32064; only the first 32000 logits are used | none |

## Licenses and attribution

My code is under the MIT license (`LICENSE`). `src/vcdpope/noise.py` is adapted from the official VCD repository (Apache-2.0, `third_party/VCD_LICENSE`). The POPE question files in `data/pope/` are copied from the same repository at commit `d6568ff`. They are the POPE splits of Li et al., EMNLP 2023. COCO images fall under the [COCO terms of use](https://cocodataset.org/#termsofuse). They are not redistributed here. The LLaVA-1.5 weights have their own license (Llama 2 community license).
