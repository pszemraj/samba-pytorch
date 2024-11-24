# samba-pytorch

> Implementation of [Samba by Microsoft](https://arxiv.org/abs/2406.07522) in PyTorch.

This aims to be a simpler implementation of the [original repo](https://github.com/microsoft/Samba).

## Installation

> [!TIP]
> The pip install command _should_ install all dependencies and the package, but some CUDA-heavy dependencies are better installed separately. See below for more details.

```bash
git clone https://github.com/pszemraj/samba-pytorch.git
cd samba-pytorch
pip install -e .
```

### Installing custom kernel packages first

After installing `torch`, `xformers`, and `flash-attn`, you may want to install `mamba-ssm`, `causal-conv1d`, and `fla` from source:

```bash
pip install --upgrade pip ninja
pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation
pip install git+https://github.com/Dao-AILab/causal-conv1d.git --no-build-isolation
pip install git+https://github.com/sustcsonglin/flash-linear-attention@98c176e --no-build-isolation
```

Then, clone this repo and run commands as above.

## Usage

A basic example of creating a random model from a named config:

```python
from samba_pytorch import Config, GPT
cfg = Config.from_name('Samba_421M_1k_window')
print*(cfg)
model = GPT(cfg)
model
```

### Training

A minimalist training script for a character-level language model on enwiki8:

```python
python examples/train.py
```

Credit to [nGPT-pytorch](https://github.com/lucidrains/nGPT-pytorch) for the enwik8 data set and the training script, which has been adapted for this repo.

## repo structure

```text
samba-pytorch/
├── pyproject.toml
├── README.md
├── samba_pytorch/
│   ├── __init__.py
│   ├── config.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── fused_rotary_embedding.py
│   │   ├── gla.py
│   │   ├── mamba_simple.py
│   │   ├── multiscale_retention.py
│   │   ├── rmsnorm.py
│   │   └── rotary.py
│   ├── samba.py
│   ├── tokenizer.py
│   └── utils.py
```

## Citations

```bibtex
@article{ren2024samba,
      title={Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling},
      author={Liliang Ren and Yang Liu and Yadong Lu and Yelong Shen and Chen Liang and Weizhu Chen},
      journal = {arXiv preprint},
      year={2024},
      url={https://arxiv.org/abs/2406.07522}
}
```
