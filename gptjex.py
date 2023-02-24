#!/usr/bin/env python3
# export PYTORCH_ENABLE_MPS_FALLBACK=1

from transformers import GPTJForCausalLM, AutoTokenizer
import torch
import os
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
print("Created model")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
print("Created tokenizer")
prompt = (
    "How do I create a thread in GoLang?"
)
tougherprompt = (
    """
function reverse_array() {
    # Usage: reverse_array "array"
    shopt -s extdebug
    f()(printf '%s\\n' "${BASH_ARGV[@]}"); f "$@"
    shopt -u extdebug
}

# Please explain how the bash shell code above works.
    """
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print("Got input_ids from tokenizer")
gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=2048,
    pad_token_id=tokenizer.eos_token_id
)
print("Generated tokens")
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print("Got gen_text from tokenizer.batch_decode\n\n\n")
print(gen_text)
os.system("say done")
