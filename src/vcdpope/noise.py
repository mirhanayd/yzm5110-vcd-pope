"""Diffusion-style Gaussian noise used to build the distorted image v'.

Adapted from the official VCD implementation
(https://github.com/DAMO-NLP-SG/VCD, file vcd_utils/vcd_add_noise.py,
commit d6568ff81b8fd306a49e630df44f2db5c2300191), Apache License 2.0.
See third_party/VCD_LICENSE. Changes: the unused variables of the original were
removed and an optional torch.Generator argument was added; the noise schedule
and the forward process q(x_t | x_0) are unchanged.
"""
import torch

NUM_STEPS = 1000


def _schedule():
    betas = torch.linspace(-6, 6, NUM_STEPS)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    return torch.sqrt(alphas_prod), torch.sqrt(1 - alphas_prod)


_ALPHAS_BAR_SQRT, _ONE_MINUS_ALPHAS_BAR_SQRT = _schedule()


def add_diffusion_noise(image_tensor: torch.Tensor, noise_step: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Return sqrt(a_bar_t) * x0 + sqrt(1 - a_bar_t) * eps, eps ~ N(0, I).

    `image_tensor` is the *normalized* pixel tensor produced by the CLIP image
    processor, exactly as in the official code (noise is added after
    normalization, not to raw pixels).
    """
    t = int(noise_step)
    if not 0 <= t < NUM_STEPS:
        raise ValueError(f"noise_step must be in [0, {NUM_STEPS - 1}], got {t}")
    noise = torch.randn(image_tensor.shape, generator=generator, dtype=image_tensor.dtype, device=image_tensor.device)
    return _ALPHAS_BAR_SQRT[t] * image_tensor + _ONE_MINUS_ALPHAS_BAR_SQRT[t] * noise
