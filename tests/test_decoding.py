"""CPU tests with a tiny random LLaVA (no GPU, ~seconds).

    python -m pytest -q tests          (or: python tests/test_decoding.py)

1. the gated decoder with gate="lossless" yields exactly the same tokens as plain VCD,
   under both sampling and greedy decoding, while skipping noisy forward passes;
2. the lazy "catch-up" of the noisy KV cache gives the same logits as a fresh prefill;
3. prompt construction expands <image> into the right number of image tokens.
"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from vcdpope.decoding import DecodeConfig, _forward, decode  # noqa: E402
from vcdpope.model import build_prompt, load_llava, tokenize_with_image  # noqa: E402
from vcdpope.noise import add_diffusion_noise  # noqa: E402

TINY = os.environ.get("TINY_LLAVA", "trl-internal-testing/tiny-LlavaForConditionalGeneration")
_LM = None


def lm():
    global _LM
    if _LM is None:
        _LM = load_llava(TINY, dtype="float32", device="cpu", attn_implementation="eager")
        torch.manual_seed(0)
        with torch.no_grad():  # make next-token distributions peaked so |keep| == 1 actually happens
            _LM.model.get_output_embeddings().weight.mul_(40.0)
    return _LM


def _inputs(seed):
    m = lm()
    torch.manual_seed(seed)
    size = m.model.config.vision_config.image_size
    pix = torch.randn(1, 3, size, size)
    pix_cd = add_diffusion_noise(pix, 500)
    ids = tokenize_with_image(build_prompt("Is there a dog in the image?"), m)
    return ids, pix, pix_cd


def test_prompt_expansion():
    m = lm()
    ids = tokenize_with_image(build_prompt("Is there a cat in the image?"), m)
    assert int((ids == m.image_token_id).sum()) == m.num_image_tokens
    assert ids[0, 0].item() == m.tokenizer.bos_token_id
    assert int((ids == m.tokenizer.bos_token_id).sum()) == 1


def test_lossless_gate_is_exact():
    m = lm()
    total_skipped = 0
    for mode in ("vcd", "vcd_greedy"):
        for beta in (0.1, 0.5):
            for seed in range(6):
                ids, pix, pix_cd = _inputs(seed)
                outs = []
                for gate in ("none", "lossless"):
                    torch.manual_seed(1000 + seed)
                    cfg = DecodeConfig(mode=mode, beta=beta, gate=gate, max_new_tokens=12, eos_token_id=-1)
                    gen, st = decode(m.model, ids, pix, pix_cd, cfg, random.Random(0))
                    outs.append((gen, st))
                (g0, s0), (g1, s1) = outs
                assert g0 == g1, (mode, beta, seed, g0, g1)
                assert s1.noisy_forwards == s0.noisy_forwards - s1.skipped_steps
                total_skipped += s1.skipped_steps
    assert total_skipped > 0, "test did not exercise any skipped step"
    print("lossless skipped steps in test:", total_skipped)


def test_noisy_catch_up_matches_fresh_prefill():
    m = lm()
    ids, pix, pix_cd = _inputs(3)
    gen = [5, 17, 42, 7]
    full = torch.cat([ids, torch.tensor([gen])], dim=1)
    fresh, _ = _forward(m.model, full, pix_cd, None)
    _, cache = _forward(m.model, ids, pix_cd, None)            # prefill on prompt only
    caught, _ = _forward(m.model, full[:, ids.shape[1]:], None, cache)  # catch up on 4 tokens at once
    assert torch.allclose(fresh, caught, atol=1e-4), (fresh - caught).abs().max()


def test_random_gate_skip_rate():
    m = lm()
    ids, pix, pix_cd = _inputs(1)
    skipped = steps = 0
    for s in range(10):
        cfg = DecodeConfig(mode="vcd_greedy", gate="random", skip_prob=0.5, max_new_tokens=10, eos_token_id=-1)
        _, st = decode(m.model, ids, pix, pix_cd, cfg, random.Random(s))
        skipped += st.skipped_steps
        steps += st.steps
    assert 0.3 < skipped / steps < 0.7


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
