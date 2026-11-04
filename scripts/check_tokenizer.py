#!/usr/bin/env python
"""Check that our prompt tokenization for the HF port matches the original LLaVA-1.5 tokenizer
(liuhaotian/llava-v1.5-7b + llava.mm_utils.tokenizer_image_token) on POPE questions."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from transformers import AutoTokenizer
from vcdpope.model import build_prompt

def tokenizer_image_token(prompt, tok):  # verbatim logic of llava/mm_utils.py, image index -200
    chunks = [tok(c).input_ids for c in prompt.split("<image>")]
    ids, offset = [], 0
    if chunks and chunks[0] and chunks[0][0] == tok.bos_token_id:
        offset = 1; ids.append(chunks[0][0])
    for i, c in enumerate(chunks):
        if i > 0: ids.append(-200)
        ids.extend(c[offset:])
    return ids

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
qs = [json.loads(l)["text"] for l in open(os.path.join(root, "data/pope/coco/coco_pope_adversarial.json"))]
orig = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
ours = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf", use_fast=True)
uniq = sorted(set(qs))
bad = [q for q in uniq if tokenizer_image_token(build_prompt(q), orig) != tokenizer_image_token(build_prompt(q), ours)]
print(f"{len(uniq) - len(bad)}/{len(uniq)} unique POPE prompts tokenize identically")
for q in bad[:5]: print("mismatch:", q)
