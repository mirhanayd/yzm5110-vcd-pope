"""Token-by-token decoding for LLaVA with Visual Contrastive Decoding (VCD) and
confidence-gated variants.

VCD (Leng et al., CVPR 2024), per step t:
    l   = logit(y | v,  x, y_<t)          clean image v
    l'  = logit(y | v', x, y_<t)          distorted image v' (diffusion noise)
    keep = { y : l_y >= log(beta) + max_w l_w }       adaptive plausibility constraint (APC)
    p_vcd = softmax( ((1+alpha) l - alpha l') masked to keep )

Observation used by the gated variants: if |keep| == 1 the masked distribution is a
one-hot on argmax(l) *whatever l' is*, so the second (noisy) forward pass cannot change
the output. ``gate="lossless"`` skips exactly those passes. ``gate="margin"`` also skips
when the clean top-token probability is >= tau (lossy), and ``gate="random"`` skips with
a fixed probability (control at matched skip rate). A skipped step falls back to the
APC-restricted clean distribution (alpha = 0 for that step).

The noisy branch keeps its own KV cache and is updated lazily: when it is needed after
one or more skipped steps it first "catches up" on the tokens it has not seen.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

from .model import REAL_VOCAB

MODES = ("regular", "greedy", "vcd", "vcd_greedy", "apc", "apc_greedy")
GATES = ("none", "lossless", "margin", "random")


@dataclass
class DecodeConfig:
    mode: str = "vcd"            # see MODES; *_greedy = argmax instead of sampling
    alpha: float = 1.0
    beta: float = 0.1
    max_new_tokens: int = 32
    gate: str = "none"           # see GATES; only used by vcd / vcd_greedy
    tau: float = 1.0             # margin gate: skip if clean top-prob >= tau
    skip_prob: float = 0.0       # random gate: skip with this probability
    eos_token_id: int = 2

    def __post_init__(self):
        assert self.mode in MODES, self.mode
        assert self.gate in GATES, self.gate

    @property
    def contrastive(self) -> bool:
        return self.mode in ("vcd", "vcd_greedy")

    @property
    def uses_apc(self) -> bool:
        return self.mode in ("vcd", "vcd_greedy", "apc", "apc_greedy")

    @property
    def sampling(self) -> bool:
        return self.mode in ("regular", "vcd", "apc")


@dataclass
class DecodeStats:
    steps: int = 0
    clean_forwards: int = 0
    noisy_forwards: int = 0
    noisy_prefills: int = 0
    skipped_steps: int = 0
    lossless_steps: int = 0      # steps where |keep| == 1 (skip would be exact)
    first: dict = field(default_factory=dict)


def _forward(model, input_ids, pixel_values, cache):
    out = model(input_ids=input_ids, pixel_values=pixel_values, past_key_values=cache, use_cache=True, return_dict=True)
    return out.logits[0, -1, :REAL_VOCAB].float(), out.past_key_values


@torch.inference_mode()
def decode(model, input_ids: torch.Tensor, pixel_values: torch.Tensor, pixel_values_cd: torch.Tensor | None,
           cfg: DecodeConfig, gate_rng: random.Random | None = None, yes_no_ids: tuple[int, int] | None = None):
    """Decode one sample (batch size 1). Returns (generated_token_ids, DecodeStats)."""
    st = DecodeStats()
    gen: list[int] = []
    clean_cache = None
    noisy_cache, noisy_seen = None, 0
    log_beta = math.log(cfg.beta) if cfg.uses_apc else None
    gate_rng = gate_rng or random.Random(0)

    for step in range(cfg.max_new_tokens):
        if clean_cache is None:
            l, clean_cache = _forward(model, input_ids, pixel_values, None)
        else:
            l, clean_cache = _forward(model, torch.tensor([[gen[-1]]], device=input_ids.device), None, clean_cache)
        st.clean_forwards += 1
        st.steps += 1

        info = {}
        if cfg.uses_apc:
            keep = l >= (log_beta + l.max())
            n_keep = int(keep.sum())
            if n_keep == 1:
                st.lossless_steps += 1
            info["n_keep"] = n_keep

        if cfg.mode in ("regular", "greedy"):
            scores = l
        elif cfg.mode in ("apc", "apc_greedy"):
            scores = l.masked_fill(~keep, -float("inf"))
        else:  # vcd / vcd_greedy
            p_top = float(torch.softmax(l, -1).max())
            if cfg.gate == "none":
                skip = False
            elif cfg.gate == "lossless":
                skip = n_keep == 1
            elif cfg.gate == "margin":
                skip = n_keep == 1 or p_top >= cfg.tau
            else:  # random
                skip = gate_rng.random() < cfg.skip_prob
            info["p_top"], info["skipped"] = p_top, skip
            if skip:
                st.skipped_steps += 1
                scores = l.masked_fill(~keep, -float("inf"))
            else:
                full = torch.tensor([gen], dtype=input_ids.dtype, device=input_ids.device)
                full = torch.cat([input_ids, full], dim=1) if gen else input_ids
                if noisy_cache is None:
                    ln, noisy_cache = _forward(model, full, pixel_values_cd, None)
                    st.noisy_prefills += 1
                else:
                    ln, noisy_cache = _forward(model, full[:, noisy_seen:], None, noisy_cache)
                noisy_seen = full.shape[1]
                st.noisy_forwards += 1
                scores = ((1 + cfg.alpha) * l - cfg.alpha * ln).masked_fill(~keep, -float("inf"))
                if step == 0 and yes_no_ids is not None:
                    info["noisy_logit_yes"], info["noisy_logit_no"] = float(ln[yes_no_ids[0]]), float(ln[yes_no_ids[1]])

        if cfg.sampling:
            tok = int(torch.multinomial(torch.softmax(scores, -1), num_samples=1))
        else:
            tok = int(scores.argmax())

        if step == 0:
            info["token"] = tok
            if yes_no_ids is not None:
                info["clean_logit_yes"], info["clean_logit_no"] = float(l[yes_no_ids[0]]), float(l[yes_no_ids[1]])
            st.first = info
        gen.append(tok)
        if tok == cfg.eos_token_id:
            break
    return gen, st
