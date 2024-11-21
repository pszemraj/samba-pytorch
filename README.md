# samba-pytorch

> Implementation of [Samba by Microsoft](https://arxiv.org/abs/2406.07522) in PyTorch.

This aims to be a simpler implementation of the [original repo](https://github.com/microsoft/Samba).

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
