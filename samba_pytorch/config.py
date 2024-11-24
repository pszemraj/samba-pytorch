# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

import torch
from typing_extensions import Self

from samba_pytorch.utils import find_multiple


@dataclass
class Config:
    """Configuration class for SAMBA (Simple Hybrid State Space Models) architecture.

    The SAMBA architecture combines Mamba (selective state space model) with
    Sliding Window Attention (SWA) and Multi-Layer Perceptrons (MLP) in a layer-wise fashion.

    Attributes:
        org (str): Organization name, defaults to "samba-pytorch"
        name (str): Model name, defaults to "lit-GPT"
        block_size (int): Maximum sequence length for the model, defaults to 4096
        vocab_size (int): Size of the vocabulary, defaults to 50254
        padding_multiple (int): Padding factor for vocab size optimization, defaults to 512
        padded_vocab_size (Optional[int]): Actual padded vocabulary size after adjustment
        n_layer (int): Number of transformer layers, defaults to 16
        n_head (int): Number of attention heads, defaults to 32
        n_embd (int): Embedding dimension / hidden state size, defaults to 4096
        rotary_percentage (float): Fraction of dimensions to apply rotary embeddings to, defaults to 0.25
        parallel_residual (bool): Whether to use parallel residual connections, defaults to True
        bias (bool): Whether to include bias terms in linear layers, defaults to True

        # SAMBA-specific parameters
        local_window (int): Size of sliding window for attention, -1 means full attention
        mlp (bool): Whether to include MLP layers, defaults to True
        full_per_layer (int): Number of tokens for full attention per layer
        mb_per_layer (int): Number of Mamba layers per block
        ret_per_layer (int): Number of RetNet layers per block
        gla_per_layer (int): Number of GLA (Gated Linear Attention) layers per block
        nope (bool): Skip certain layers if True
        mamba (bool): Whether to use Mamba layers, defaults to False
        sc_attn (bool): Whether to use short convolution in attention, defaults to False
        rms_norm (bool): Use RMSNorm instead of LayerNorm, defaults to True

        # Performance optimizations
        residual_in_fp32 (bool): Keep residual connections in fp32, defaults to True
        fused_add_norm (bool): Use fused add+norm operations, defaults to True
        mamba_init (bool): Use specialized Mamba initialization, defaults to False
        attn_layer_pos (str): Position of attention layers in architecture
        n_query_groups (Optional[int]): Number of query groups for grouped-query attention
        shared_attention_norm (bool): Share normalization across attention heads, defaults to False

        _norm_class (str): Normalization layer class to use ("LayerNorm" or "RMSNorm")
        norm_eps (float): Epsilon for normalization layers, defaults to 1e-5
        _mlp_class (str): MLP implementation class ("GptNeoxMLP" or "LLaMAMLP")
        intermediate_size (Optional[int]): Size of intermediate MLP layers
        condense_ratio (int): Ratio for condensing layers, defaults to 1

    Key Implementation Details from Paper:
    - SAMBA combines Mamba, SWA and MLP through layer-wise interleaving
    - Default sliding window size is 2048 tokens
    - Uses PreNorm and skip connections for each intermediate layer
    - Mamba layers capture time-dependent semantics and provide efficient decoding
    - SWA handles complex non-Markovian dependencies
    - MLPs handle factual knowledge recall
    """

    org: str = "samba-pytorch"
    name: str = "lit-GPT"
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    local_window: int = -1
    mlp: bool = True
    full_per_layer: int = 1000000
    mb_per_layer: int = -1
    ret_per_layer: int = -1
    gla_per_layer: int = -1
    nope: bool = False
    mamba: bool = False
    sc_attn: bool = False
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    mamba_init: bool = False
    attn_layer_pos: str = None
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"] = "GptNeoxMLP"
    intermediate_size: Optional[int] = None
    condense_ratio: int = 1

    def __post_init__(self):
        # error checking
        assert self.n_embd % self.n_head == 0
        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(
                self.vocab_size, self.padding_multiple
            )
        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @property
    def mlp_class(self) -> Type:
        from samba_pytorch import samba

        # `self._mlp_class` cannot be the type to keep the config json serializable
        return getattr(samba, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":
            from samba_pytorch.modules.rmsnorm import RMSNorm

            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            warnings.warn(
                "FusedRMSNorm has been removed, using standard torch RMSNorm instead"
            )
            from samba_pytorch.modules.rmsnorm import RMSNorm

            return RMSNorm
        return getattr(torch.nn, self._norm_class)


configs = []

Samba = [
    dict(
        org="Microsoft",
        name="LLaMA_438M",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_438M_SWA",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        local_window=2048,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_438M_SWA_full",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        full_per_layer=2,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        local_window=2048,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_438M_SWA_mqa",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4608,
        local_window=2048,
        n_query_groups=1,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_438M_SWA_mqa6",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=6,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4608,
        local_window=2048,
        n_query_groups=1,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_438M_SWA_gqa2",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4608,
        local_window=2048,
        n_query_groups=2,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_438M_SWA_gqa4",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4608,
        local_window=2048,
        n_query_groups=4,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_438M2k",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_438M8k_SWA",
        block_size=8192,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        local_window=2048,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_438M16k_SWA",
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        local_window=2048,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_438M32k_SWA",
        block_size=32768,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        local_window=2048,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_438M_SWA_paral",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=14,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=True,
        shared_attention_norm=True,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_438M_SWA_8k4k",
        block_size=8192,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=14,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        local_window=4096,
        mamba_init=True,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_438M_SWA_8k2k",
        block_size=8192,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=14,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="Microsoft",
        name="Samba_421M",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        mb_per_layer=2,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_449M_full_mamba_0",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        mamba_init=True,
        attn_layer_pos="[0,]",
    ),
    dict(
        org="Microsoft",
        name="LLaMA_449M_full_mamba_11",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        mamba_init=True,
        attn_layer_pos="[11,]",
    ),
    dict(
        org="Microsoft",
        name="LLaMA_449M_full_mamba_5",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        mamba_init=True,
        attn_layer_pos="[5,]",
    ),
    dict(
        org="Microsoft",
        name="LLaMA_443M_full_mamba_1,5",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        mamba_init=True,
        attn_layer_pos="[1,5]",
    ),
    dict(
        org="Microsoft",
        name="Samba_421M_mqa",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        mb_per_layer=2,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4608,
        local_window=2048,
        mamba_init=True,
        n_query_groups=1,
    ),
    dict(
        org="Microsoft",
        name="Samba_421M_gqa4",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        mb_per_layer=2,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4608,
        local_window=2048,
        mamba_init=True,
        n_query_groups=4,
    ),
    dict(
        org="Microsoft",
        name="Samba_421M_gqa2",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        mb_per_layer=2,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4608,
        local_window=2048,
        mamba_init=True,
        n_query_groups=2,
    ),
    dict(
        org="Microsoft",
        name="Samba_421M_mqa6",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        mb_per_layer=2,
        n_layer=12,
        n_head=6,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4608,
        local_window=2048,
        mamba_init=True,
        n_query_groups=1,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_438M_SWA_gla",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gla_per_layer=2,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        local_window=2048,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_446M_SWA_ret",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        ret_per_layer=2,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="Microsoft",
        name="Samba_421M_1k_window",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        mb_per_layer=2,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        local_window=1024,
        mamba_init=True,
    ),
    dict(
        org="Microsoft",
        name="Samba_421M_512_window",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        mb_per_layer=2,
        n_layer=12,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        local_window=512,
        mamba_init=True,
    ),
    ###############  1.3B   ###############
    dict(
        org="Microsoft",
        name="Samba_1.3B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        mb_per_layer=2,
        n_layer=18,
        n_head=18,
        n_embd=2304,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_1.2B_SWA_gla",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gla_per_layer=2,
        n_layer=18,
        n_head=18,
        n_embd=2304,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_1.4B_SWA_ret",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        ret_per_layer=2,
        n_layer=18,
        n_head=18,
        n_embd=2304,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_1.3B_SWA",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=20,
        n_head=18,
        n_embd=2304,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_1.3B_SWA_8k4k",
        block_size=8192,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=20,
        n_head=18,
        n_embd=2304,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window=4096,
        mamba_init=True,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_1.3B_SWA_8k2k",
        block_size=8192,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=20,
        n_head=18,
        n_embd=2304,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_1.3B",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=20,
        n_head=18,
        n_embd=2304,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        mamba_init=True,
    ),
    dict(
        org="Microsoft",
        name="LLaMA_2.9B",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=24,
        n_head=24,
        n_embd=3072,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=8192,
    ),
]
configs.extend(Samba)

mamba = [
    dict(
        name="Mamba_370M",
        mamba=True,
        block_size=4096,
        padding_multiple=64,
        n_layer=48,
        vocab_size=32000,
        n_embd=1024,
        norm_eps=1e-5,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
    ),
    dict(
        name="Mamba_430M",
        mamba=True,
        block_size=4096,
        padding_multiple=64,
        n_layer=60,
        vocab_size=32000,
        n_embd=1024,
        norm_eps=1e-5,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
    ),
    dict(
        name="Mamba_1.3B",
        mamba=True,
        block_size=4096,
        padding_multiple=64,
        n_layer=48,
        vocab_size=32000,
        n_embd=2048,
        norm_eps=1e-5,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
    ),
]
configs.extend(mamba)

name_to_config = {config["name"]: config for config in configs}
