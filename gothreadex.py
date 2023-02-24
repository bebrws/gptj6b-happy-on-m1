#!/usr/bin/env python3
# Default: THIS WORKS
# NEDS:
# export PYTORCH_ENABLE_MPS_FALLBACK=1

from transformers import GPTJForCausalLM, AutoTokenizer
import torch
import os
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
prompt = (
    "How do I create a thread in GoLang?"
)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=4096,
    pad_token_id=tokenizer.eos_token_id
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)
os.system("say done")
