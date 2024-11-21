import torch
from torch import nn
from einops import rearrange
from typing import Optional, Tuple, Union


def maybe_align(x: torch.Tensor, alignment_in_bytes: int = 16) -> torch.Tensor:
    """Ensures memory alignment by cloning if necessary."""
    return x if x.data_ptr() % alignment_in_bytes == 0 else x.clone()


def dropout_add_layer_norm(
    x0: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    dropout_p: float,
    epsilon: float,
    rowscale: Optional[torch.Tensor] = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
    return_dropout_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Fused dropout + residual add + layer norm implementation.

    Args:
        x0: Input tensor
        residual: Optional residual tensor to add
        weight: Layer norm weight parameter
        bias: Optional layer norm bias parameter
        dropout_p: Dropout probability
        epsilon: Small constant for numerical stability
        rowscale: Optional row-wise scaling factor
        prenorm: Whether to return pre-normalization results
        residual_in_fp32: Whether to cast residual to fp32 during addition
        is_rms_norm: Whether to use RMS normalization instead of layer norm
        return_dropout_mask: Whether to return the dropout mask
    """
    # Initialize mask
    mask = None

    # Apply dropout
    if dropout_p > 0.0:
        mask = torch.bernoulli(torch.full_like(x0, 1 - dropout_p))
        x0 = x0 * mask / (1 - dropout_p)

    # Add residual if provided
    if residual is not None:
        if residual_in_fp32:
            x0 = x0 + residual.float().to(x0.dtype)
        else:
            x0 = x0 + residual

    # Apply row scaling if provided
    if rowscale is not None:
        x0 = x0 * rearrange(rowscale, "b -> b 1")

    # Compute normalization (either LayerNorm or RMSNorm)
    if is_rms_norm:
        norm_x = torch.mean(x0 * x0, dim=-1, keepdim=True)
        x_normed = x0 * torch.rsqrt(norm_x + epsilon)
    else:
        mean = x0.mean(dim=-1, keepdim=True)
        var = x0.var(dim=-1, unbiased=False, keepdim=True)
        x_normed = (x0 - mean) / torch.sqrt(var + epsilon)

    # Apply weight and optional bias
    output = x_normed * weight + (bias if bias is not None else 0.0)

    if return_dropout_mask:
        if mask is None:
            mask = torch.ones_like(x0, dtype=torch.uint8)
        return output, mask
    return output


class DropoutAddLayerNorm(nn.Module):
    """
    Module that combines dropout, residual connection, and layer normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        prenorm: bool = False,
        p: float = 0.0,
        eps: float = 1e-5,
        residual_in_fp32: bool = False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.prenorm = prenorm
        self.p = p
        self.eps = eps
        self.residual_in_fp32 = residual_in_fp32
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(hidden_size, **factory_kwargs))

    def forward(
        self,
        x0: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        rowscale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return dropout_add_layer_norm(
            x0,
            residual,
            self.weight,
            self.bias,
            self.p if self.training else 0.0,
            self.eps,
            rowscale=rowscale,
            prenorm=self.prenorm,
            residual_in_fp32=self.residual_in_fp32,
        )

    def reset_parameters(self):
        """Reset parameters to default initialization."""
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Implementation follows the paper: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)

    def reset_parameters(self):
        """Reset parameters to default initialization."""
        nn.init.ones_(self.weight)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Applies RMS normalization to the input tensor.
    """
    norm_x = torch.mean(x * x, dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(norm_x + epsilon)
    return x_normed * weight
