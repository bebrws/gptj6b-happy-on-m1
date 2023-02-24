# Default: THIS WORKS
# NEDS:
# export PYTORCH_ENABLE_MPS_FALLBACK=1

from transformers import GPTJForCausalLM, AutoTokenizer
import torch
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
    pad_token_id=tokenizer.eos_token_id
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)


# Float 64

# from transformers import GPTJForCausalLM, AutoTokenizer
# import torch
# model = GPTJForCausalLM.from_pretrained(
#     "EleutherAI/gpt-j-6B", torch_dtype=torch.float64)
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
# prompt = (
#     "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
#     "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
#     "researchers was the fact that the unicorns spoke perfect English."
# )
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# gen_tokens = model.generate(
#     input_ids,
#     do_sample=True,
#     temperature=0.9,
#     max_length=100,
#     pad_token_id=tokenizer.eos_token_id
# )
# gen_text = tokenizer.batch_decode(gen_tokens)[0]
# print(gen_text)

# FLOAT 16:


# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import GPTJForCausalLM
# import torch
# model = GPTJForCausalLM.from_pretrained(
#     "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
# )
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# prompt = (
#     "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
#     "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
#     "researchers was the fact that the unicorns spoke perfect English."
# )

# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# gen_tokens = model.generate(
#     input_ids,
#     do_sample=True,
#     temperature=0.9,
#     max_length=100,
# )

# gen_text = tokenizer.batch_decode(gen_tokens)[0]
