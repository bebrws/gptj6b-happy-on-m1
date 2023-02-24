# Got it working!!

## Following:
https://lazycoder.ro/posts/using-gpt-neo-on-m1-mac/
And
https://github.com/huggingface/transformers/blob/ba0e370dc1713b0ddd9b1be0ac31ef1fdc7bdf76/docs/source/en/model_doc/gptj.mdx?plain=1#L62

## Setup:
PreRequirements:
* Install xcode and brew
* MiniForge3 (don’t use Anaconda) -> download the arm64 sh from GitHub - https://github.com/conda-forge/miniforge#download
* Make sure you have the latest rust installed via rustup

Steps:

```
# set up conda for fish
~/miniforge3/bin/conda init zsh

# create and use a new env
conda create --name tf python=3.9
conda activate tf

# install tensorflow deps
conda install -c apple tensorflow-deps
# base tensorflow + metal plugin
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal

# install jupyter, pandas and whatnot
conda install -c conda-forge -y pandas jupyter

mkdir -p Projects/lab/tfsetup && cd Projects/lab/tfsetup
git clone https://github.com/huggingface/tokenizers
cd tokenizers/bindings/python
# compile tokenizers - should be pretty fast on your m1
pip install setuptools_rust
# install tokenizers
python setup.py install

# install transformers using pip
pip install git+https://github.com/huggingface/transformers
pip install numpy --upgrade --ignore-installed

arch -arm64 brew install cmake
arch -arm64 brew install pkgconfig


cd ../../../../../../ # Back to root dir
```

If you run now you will see errors about MPS not supporting different things:
```
RuntimeError: MPS does not support cumsum op with int64 input
```

To get around this:
```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Testing install

```
import tensorflow as tf

print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

Run the actual code!!!:
```
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
```

