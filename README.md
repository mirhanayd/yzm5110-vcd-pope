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

```bash
bash scripts/experiments.sh smoke     # 100 questions x 3 configs, checks the pipeline
bash scripts/experiments.sh phase1    # dev gate tuning (A-OKVQA POPE, images disjoint from COCO), then
                                      # COCO random|popular|adversarial: Regular / VCD / APC-only x 3 seeds, greedy / VCD-greedy
TAU=... SKIP=... bash scripts/experiments.sh phase2   # lossless / margin / random gate on COCO x 3 seeds + timing runs alone on the GPU
bash scripts/experiments.sh tables
```

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
| noise step T | code default 500; paper defers to the supplement | 500 | sensitivity can be checked with `--noise-step` |
| runs | 5 runs, mean ± std | 3 seeds (compute budget) | wider uncertainty |
| seeding | one global seed per run | seed per (run seed, question id) | resumable runs, paired comparisons across modes |
| max new tokens | 1024 | 32 (answers are one word, then EOS) | none expected on POPE |
| vocabulary | 32000 | HF port pads the LM head to 32064; we use the first 32000 logits | none |

## Licenses and attribution

Our code is MIT-licensed (`LICENSE`). `src/vcdpope/noise.py` is adapted from the official VCD repository (Apache-2.0, `third_party/VCD_LICENSE`). POPE question files (`data/pope/`) are copied from the VCD repository at commit `d6568ff` (Apache-2.0); they are the POPE splits of Li et al., EMNLP 2023. COCO images are subject to the [COCO terms of use](https://cocodataset.org/#termsofuse) and are not redistributed here. LLaVA-1.5 weights are subject to their own license (Llama 2 community license).

## AI assistance

Disclosed per the course policy: Claude (Anthropic) was used to help with code scaffolding, the analysis of the official implementation, and editing. All experiments were run and checked by the author. See the report for details.
