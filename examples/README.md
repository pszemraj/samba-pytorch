# Training examples

## Training with a tokenizer

first, install the required packages:

```bash
pip install -r examples/requirements.txt
```

then, get/save a tokenizer file to use (either `.json`  or `.model`). You can use any tokenizer, but since the example model is small, it makes sense to use a small tokenizer. There are many options from [the SAIL tokenizer study](https://huggingface.co/sail/scaling-with-vocab-trained-tokenizers).

Here we get one with a 16k vocab size:

```bash
wget -P sail-16k https://huggingface.co/sail/scaling-with-vocab-trained-tokenizers/resolve/main/hf_slimpajama-6B-16384-BPE/tokenizer.model
```

then, run the training script:

```bash
python examples/train_tokenizer.py ./sail-16k
```
