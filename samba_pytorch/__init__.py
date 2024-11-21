"""Minimal implementation of Samba by Microsoft in PyTorch."""

from samba_pytorch.config import Config
from samba_pytorch.samba import GPT, Block, CausalSelfAttention, LLaMAMLP
from samba_pytorch.tokenizer import Tokenizer
from samba_pytorch.utils import (
    chunked_cross_entropy,
    get_default_supported_precision,
    lazy_load,
    num_parameters,
)

try:
    from samba_pytorch._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    "Config",
    "GPT",
    "Block",
    "CausalSelfAttention",
    "LLaMAMLP",
    "Tokenizer",
    "chunked_cross_entropy",
    "get_default_supported_precision",
    "lazy_load",
    "num_parameters",
    "__version__",
]
