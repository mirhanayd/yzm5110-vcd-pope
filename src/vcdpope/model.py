"""Loading LLaVA-1.5 (Hugging Face port) and building inputs that mirror the
official LLaVA / VCD POPE evaluation script.

Reference behaviour (VCD repo, experiments/eval/object_hallucination_vqa_llava.py):
  * conversation template ``llava_v1`` (vicuna_v1 system prompt)
  * question + " Please answer this question with one word."
  * ``image_processor.preprocess(image)`` -> CLIP resize(shortest=336) + center crop 336,
    i.e. *no* pad-to-square, and noise is added to the normalized tensor.
  * ``tokenizer_image_token``: every text chunk is tokenized separately and the BOS of
    every chunk after the first is dropped; the image placeholder sits between chunks.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
POPE_SUFFIX = " Please answer this question with one word."
REAL_VOCAB = 32000  # LLaMA/Vicuna vocabulary; the HF port pads the LM head to 32064 rows


def build_prompt(question: str) -> str:
    # SeparatorStyle.TWO with sep=" ", messages = [USER: "<image>\n" + q, ASSISTANT: None]
    return f"{SYSTEM} USER: <image>\n{question}{POPE_SUFFIX} ASSISTANT:"


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    image_processor: object
    image_token_id: int
    num_image_tokens: int
    device: torch.device
    dtype: torch.dtype


def load_llava(model_id: str = "llava-hf/llava-1.5-7b-hf", dtype: str = "float16", load_in_4bit: bool = False,
               device: str | None = None, attn_implementation: str = "sdpa") -> LoadedModel:
    from transformers import AutoImageProcessor, AutoTokenizer, LlavaForConditionalGeneration

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    kwargs = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation=attn_implementation)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype,
                                                           bnb_4bit_quant_type="nf4")
        kwargs["device_map"] = {"": device.index or 0}
    model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if not load_in_4bit:
        model.to(device)
    model.eval()
    # The official code uses the slow LLaMA tokenizer of liuhaotian/llava-v1.5-7b. For the HF port
    # the *fast* tokenizer reproduces its ids exactly on the POPE prompt, while the HF slow
    # tokenizer (legacy=False) drops the "▁" before "\n" (checked: scripts/check_tokenizer.py).
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    image_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)  # slow CLIP processor, as in the original code
    vc = model.config.vision_config
    n_img = (vc.image_size // vc.patch_size) ** 2  # 576 for CLIP ViT-L/14-336 ("default" strategy drops CLS)
    if getattr(model.config, "vision_feature_select_strategy", "default") == "full":
        n_img += 1
    return LoadedModel(model, tokenizer, image_processor, model.config.image_token_index, n_img, device, torch_dtype)


def tokenize_with_image(prompt: str, lm: LoadedModel) -> torch.Tensor:
    """Re-implementation of LLaVA's ``tokenizer_image_token`` for the HF port.

    The single ``<image>`` placeholder is expanded to ``num_image_tokens`` copies of the
    image token id, which the HF model replaces with the projected CLIP features.
    """
    tok = lm.tokenizer
    chunks = [tok(c).input_ids for c in prompt.split("<image>")]
    ids: list[int] = []
    offset = 0
    if chunks and chunks[0] and chunks[0][0] == tok.bos_token_id:
        offset = 1
        ids.append(chunks[0][0])
    for i, ch in enumerate(chunks):
        if i > 0:
            ids.extend([lm.image_token_id] * lm.num_image_tokens)
        ids.extend(ch[offset:])
    return torch.tensor([ids], dtype=torch.long)


def preprocess_image(path: str, lm: LoadedModel) -> torch.Tensor:
    image = Image.open(path)
    # CLIPImageProcessor converts to RGB, resizes shortest edge to 336, center-crops 336, normalizes.
    return lm.image_processor(image, return_tensors="pt")["pixel_values"]  # (1, 3, 336, 336), float32, CPU
