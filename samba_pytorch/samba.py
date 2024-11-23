# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import math
import warnings
from functools import partial
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor
from typing_extensions import Self
from xformers.ops import SwiGLU

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from causal_conv1d import causal_conv1d_fn
from einops import rearrange

warnings.filterwarnings("ignore", category=FutureWarning, module="fla.ops")

from samba_pytorch.config import Config  # noqa
from samba_pytorch.modules.gla import GatedLinearAttention  # noqa
from samba_pytorch.modules.mamba_simple import Mamba  # noqa
from samba_pytorch.modules.multiscale_retention import MultiScaleRetention  # noqa

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]


def create_block(
    d_model,
    rotary_emb,  # Added rotary_emb parameter
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(
        Mamba, layer_idx=layer_idx, rotary_emb=rotary_emb, **ssm_cfg, **factory_kwargs
    )  # Passed rotary_emb
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = MBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        rotary_emb=rotary_emb,  # Passed rotary_emb
    )
    block.layer_idx = layer_idx
    return block


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        factory_kwargs = {"device": "cuda", "dtype": torch.float32}
        assert config.padded_vocab_size is not None
        self.config = config

        self.rotary_emb = RotaryEmbedding(
            dim=int(config.rotary_percentage * config.head_size),  # TODO: validate
            use_xpos=getattr(config, "use_xpos", False),
            interpolate_factor=getattr(config, "interpolate_factor", 1.0),
        )

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        if config.mamba:
            if self.config.fused_add_norm:
                if layer_norm_fn is None or rms_norm_fn is None:
                    raise ImportError(
                        "Failed to import Triton LayerNorm / RMSNorm kernels"
                    )

            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                    h=nn.ModuleList(
                        create_block(
                            config.n_embd,
                            rotary_emb=self.rotary_emb,  # TODO: is this needed?
                            ssm_cfg=None,
                            norm_epsilon=config.norm_eps,
                            rms_norm=config.rms_norm,
                            residual_in_fp32=config.residual_in_fp32,
                            fused_add_norm=config.fused_add_norm,
                            layer_idx=i,
                            **factory_kwargs,
                        )
                        for i in range(config.n_layer)
                    ),
                    ln_f=(nn.LayerNorm if not config.rms_norm else RMSNorm)(
                        config.n_embd,
                        eps=config.norm_eps,
                        **factory_kwargs,
                    ),
                )
            )

        else:
            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                    h=nn.ModuleList(
                        Block(config, i, self.rotary_emb) for i in range(config.n_layer)
                    ),
                    ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
                )
            )

        self.kv_caches: List[KVCache] = []
        self.max_len = self.config.block_size
        self.mamba_init = config.mamba or config.mamba_init
        if self.mamba_init:
            self.tie_weights()

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            if self.mamba_init:
                torch.nn.init.normal_(module.weight, std=0.02)
            else:
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd)
                )
        elif isinstance(module, nn.Linear):
            if self.mamba_init:
                if module.bias is not None:
                    if not getattr(module.bias, "_no_reinit", False):
                        nn.init.zeros_(module.bias)
            else:
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd)
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        # GPT-NeoX
        for name, p in module.named_parameters():
            if (
                name in ["out_proj.weight", "fc2.weight"]
                or (name == "proj.weight" and isinstance(module, LLaMAMLP))
                or (
                    name == "w3.weight"
                    and isinstance(module, SwiGLU)
                    or (
                        name == "proj.weight"
                        and isinstance(module, CausalSelfAttention)
                    )
                )
            ):  # if use xformer swiglu, fc2 layer will be renamed to w3
                if self.mamba_init:
                    n_residuals_per_layer = (
                        1 if self.config.mamba or not self.config.mlp else 2
                    )
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)
                else:
                    nn.init.normal_(
                        p, mean=0.0, std=1 / math.sqrt(self.config.n_embd) / n_layer
                    )

    def tie_weights(self):
        self.lm_head.weight = self.transformer.wte.weight

    def reset_cache(self) -> None:
        self.max_len = self.config.block_size
        self.kv_caches.clear()
        # (Removed cache reset for rope_cache and mask_cache)

    def forward(
        self,
        idx: torch.Tensor,
        max_seq_length: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.mamba:
            hidden_states = self.transformer.wte(idx)
            residual = None
            for block in self.transformer.h:
                hidden_states, residual = block(
                    hidden_states, residual, inference_params=None
                )
            norm_f = self.transformer.ln_f
            if not self.config.fused_add_norm:
                residual = (
                    (hidden_states + residual)
                    if residual is not None
                    else hidden_states
                )
                hidden_states = norm_f(residual.to(dtype=norm_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = (
                    rms_norm_fn if isinstance(norm_f, RMSNorm) else layer_norm_fn
                )
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    norm_f.weight,
                    norm_f.bias,
                    eps=norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.config.residual_in_fp32,
                )
            return self.lm_head(hidden_states)

        B, T = idx.size()
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:  # not relevant otherwise
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"

        # Similarly, mask_cache is removed. TODO: validate

        # Create mask if using kv_cache
        if use_kv_cache:
            mask = self.build_mask_cache(idx).index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            mask = None

        # Initialize rotary embedding variables
        if self.config.nope:
            rope = None  # Set rope to None if config.nope
        else:
            # Using rotary_emb to rotate queries and keys in attention modules
            rope = self.rotary_emb

        # Forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if not use_kv_cache:
            for block in self.transformer.h:
                x, *_ = block(x, rope, max_seq_length)
        else:
            if self.config.nope:
                self.kv_caches = self.kv_caches or self.build_kv_caches(
                    x, max_seq_length, None
                )
            else:
                # rotary_emb handles the rotations
                self.kv_caches = self.kv_caches or self.build_kv_caches(
                    x,
                    max_seq_length,
                    None,  # rotary_emb handle offsets
                )
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(
                    x, rope, max_seq_length, mask, input_pos, self.kv_caches[i]
                )

        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(
            (self.config.block_size, self.config.block_size),
            device=idx.device,
            dtype=torch.bool,
        )
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(
        self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: int
    ) -> List[KVCache]:
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups
        # rotary_emb handles the rope_cache_length, thus set to None or appropriately
        k_cache_shape = (
            B,
            max_seq_length,
            heads,
            self.config.head_size,
        )
        v_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        device = idx.device
        return [
            (
                torch.zeros(k_cache_shape, device=device),
                torch.zeros(v_cache_shape, device=device),
            )
            for _ in range(self.config.n_layer)
        ]


class Block(nn.Module):
    def __init__(
        self, config: Config, layer_idx: int, rotary_emb: RotaryEmbedding
    ) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        if config.attn_layer_pos is not None:
            self.use_mamba = layer_idx not in eval(config.attn_layer_pos)
        else:
            self.use_mamba = (
                layer_idx % config.mb_per_layer == 0
                if config.mb_per_layer > 0
                else False
            )
        self.use_retnet = (
            layer_idx % config.ret_per_layer == 0 if config.ret_per_layer > 0 else False
        )
        self.use_gla = (
            layer_idx % config.gla_per_layer == 0 if config.gla_per_layer > 0 else False
        )
        if self.use_mamba:
            factory_kwargs = {"device": "cuda", "dtype": torch.float32}
            self.attn = Mamba(config.n_embd, layer_idx=layer_idx, **factory_kwargs)
        elif self.use_retnet:
            self.attn = MultiScaleRetention(
                hidden_size=config.n_embd,
                num_heads=config.n_head // 2,
                expand_k=1,
                expand_v=2,
                mode="fused_chunk",
                use_short_conv=False,
            )
        elif self.use_gla:
            self.attn = GatedLinearAttention(
                hidden_size=config.n_embd,
                num_heads=config.n_embd // 384,
                expand_k=0.5,
                expand_v=1,
                mode="fused_chunk",
                use_short_conv=False,
            )
        else:
            self.attn = CausalSelfAttention(
                config,
                n_embd=config.n_embd,
                layer_idx=layer_idx,
                rotary_emb=rotary_emb,
            )

        if (
            not config.shared_attention_norm
            and config.mlp
            and not config.parallel_residual
        ):
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        if config.mlp:
            self.mlp = config.mlp_class(
                config,
            )
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: Optional[RotaryEmbedding],
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        n_1 = self.norm_1(x)

        if self.use_mamba:
            h = self.attn(n_1, inference_params=kv_cache)
            new_kv_cache = kv_cache  # TODO
            x = x.to(torch.float32)
        elif self.use_retnet or self.use_gla:
            h, _, new_kv_cache = self.attn(n_1)
        else:
            h, new_kv_cache = self.attn(
                n_1, rotary_emb, max_seq_length, mask, input_pos, kv_cache
            )
        if self.config.parallel_residual:
            assert self.config.shared_attention_norm
            if self.config.mlp:
                h = h + self.mlp(n_1)
            x = x + h
        else:
            x = x + h
            if self.config.mlp:
                n_2 = self.norm_2(x)
                h = self.mlp(n_2)
                x = x + h
        return x, new_kv_cache


class MBlock(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config: Config,
        layer_idx: int,
        n_embd: int,
        rotary_emb: RotaryEmbedding,
        head_size=None,
    ) -> None:
        super().__init__()
        self.rotary_emb = rotary_emb  # Store rotary_emb
        self.local = layer_idx % config.full_per_layer < config.full_per_layer - 1
        if head_size is not None:
            self.head_size = head_size
            self.n_head = n_embd // head_size
            self.n_query_groups = self.n_head
        else:
            self.head_size = config.head_size
            self.n_head = config.n_head
            self.n_query_groups = config.n_query_groups
        shape = (self.n_head + 2 * self.n_query_groups) * self.head_size
        # Key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(n_embd, shape, bias=config.bias)
        # Output projection
        self.proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.config = config
        self.sc = config.sc_attn
        if self.sc:
            self.q_dim = self.n_head * self.head_size
            self.kv_dim = self.n_query_groups * self.head_size
            d_conv = 4
            self.q_conv1d = nn.Conv1d(
                in_channels=self.q_dim,
                out_channels=self.q_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.q_dim,
                padding=d_conv - 1,
            )
            self.k_conv1d = nn.Conv1d(
                in_channels=self.kv_dim,
                out_channels=self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.kv_dim,
                padding=d_conv - 1,
            )
            self.v_conv1d = nn.Conv1d(
                in_channels=self.kv_dim,
                out_channels=self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.kv_dim,
                padding=d_conv - 1,
            )

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: Optional[RotaryEmbedding],
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = (
            x.size()
        )  # Batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)
        # Assemble into a number of query groups to support MHA, MQA, and GQA together
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2  # Each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(
            B, T, self.n_query_groups, total_qkv, self.head_size
        )  # (B, T, n_query_groups, total_qkv, hs)

        # Split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B, T, -1)  # (B, T, nh_q, hs)
        k = k.reshape(B, T, -1)
        v = v.reshape(B, T, -1)
        if self.sc:
            q = causal_conv1d_fn(
                x=q.transpose(-1, -2),
                weight=rearrange(self.q_conv1d.weight, "d 1 w -> d w"),
                bias=self.q_conv1d.bias,
                activation="silu",
            ).transpose(-1, -2)
            k = causal_conv1d_fn(
                x=k.transpose(-1, -2),
                weight=rearrange(self.k_conv1d.weight, "d 1 w -> d w"),
                bias=self.k_conv1d.bias,
                activation="silu",
            ).transpose(-1, -2)
            v = causal_conv1d_fn(
                x=v.transpose(-1, -2),
                weight=rearrange(self.v_conv1d.weight, "d 1 w -> d w"),
                bias=self.v_conv1d.bias,
                activation="silu",
            ).transpose(-1, -2)

        q = q.reshape(B, T, -1, self.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B, T, -1, self.head_size)
        v = v.reshape(B, T, -1, self.head_size)

        if not self.config.nope and rotary_emb is not None:
            # Apply rotary embeddings using rotary-embedding-torch
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # Check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # Shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)

            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, -1)  # Re-assemble all head outputs side by side

        # Output projection
        y = self.proj(y)
        return y, kv_cache

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        scale = 1.0 / math.sqrt(self.head_size)

        if (
            mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            if self.local and self.config.local_window > -1:
                win_tuple = (self.config.local_window - 1, 0)
            else:
                win_tuple = (-1, -1)
            return flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=True,
                window_size=win_tuple,
            )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)


class LLaMAMLP(nn.Module):
    def __init__(
        self,
        config: Config,
    ) -> None:
        super().__init__()
        self.swiglu = SwiGLU(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            _pack_weights=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.swiglu(x)
        return x
