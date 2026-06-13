# References:
# https://github.com/hpcaitech/Open-Sora
# https://github.com/facebookresearch/DiT/blob/main/models.py
# https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
# https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py#L14

from contextlib import nullcontext
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp

from typing import Optional


class FeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm_FP32(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LayerNorm_FP32(nn.LayerNorm):
    def __init__(self, dim, eps, elementwise_affine):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float() ,
            self.eps
        ).to(origin_dtype)
        return out


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        B, C, T, H, W = x.shape
        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x


def modulate_fp32(norm_func, x, shift, scale):
    # Suppose x is (B, N, D), shift is (B, -1, D), scale is (B, -1, D)
    # ensure the modulation params be fp32
    assert shift.dtype == torch.float32, scale.dtype == torch.float32
    dtype = x.dtype
    x = norm_func(x.to(torch.float32))
    x = x * (scale + 1) + shift
    x = x.to(dtype)
    return x


MPS_MODULATION_MAX_CHUNK_TOKENS = 16


def _normalize_dim(dim, rank):
    if rank <= 0:
        raise ValueError("chunked modulation requires a tensor with at least one dimension.")
    normalized = dim + rank if dim < 0 else dim
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"chunk_dim {dim} is out of range for tensor rank {rank}.")
    return normalized


def _modulation_param_slice(value, *, dim, start, length, target_size):
    if value.shape[dim] == target_size:
        return value.narrow(dim, start, length)
    if value.shape[dim] == 1:
        return value
    raise ValueError(
        "modulation parameter shape is not broadcast-compatible with chunked input: "
        f"dim={dim}, parameter_size={value.shape[dim]}, target_size={target_size}"
    )


def modulate_fp32_chunked(
    norm_func,
    x,
    shift,
    scale,
    *,
    chunk_dim=-2,
    max_chunk_tokens=MPS_MODULATION_MAX_CHUNK_TOKENS,
):
    # IMPORTANT: MPS cannot afford whole-activation fp32 promotion for Avatar DiT.
    # Chunk only token/time dimensions; LayerNorm must still see the full hidden C dimension.
    assert shift.dtype == torch.float32, scale.dtype == torch.float32
    rank = x.dim()
    dim = _normalize_dim(chunk_dim, rank)
    if dim == rank - 1:
        raise ValueError("chunked fp32 modulation must not split the hidden normalization dimension.")
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive.")

    target_size = x.shape[dim]
    if target_size <= max_chunk_tokens:
        return modulate_fp32(norm_func, x, shift, scale)

    dtype = x.dtype
    out = torch.empty_like(x)
    for start in range(0, target_size, max_chunk_tokens):
        length = min(max_chunk_tokens, target_size - start)
        x_slice = x.narrow(dim, start, length)
        shift_slice = _modulation_param_slice(shift, dim=dim, start=start, length=length, target_size=target_size)
        scale_slice = _modulation_param_slice(scale, dim=dim, start=start, length=length, target_size=target_size)
        chunk = norm_func(x_slice.to(torch.float32))
        chunk = chunk * (scale_slice + 1) + shift_slice
        out.narrow(dim, start, length).copy_(chunk.to(dtype))
    return out


def modulate_mps_low_memory_chunked(
    norm_func,
    x,
    shift,
    scale,
    *,
    chunk_dim=-2,
    max_chunk_tokens=MPS_MODULATION_MAX_CHUNK_TOKENS,
):
    # IMPORTANT: On 16 GB MPS hosts, even per-slice activation fp32 casts can
    # stall or OOM. Keep normalization chunked, then keep modulation in the
    # activation dtype for this branch-local MPS fallback.
    assert shift.dtype == torch.float32, scale.dtype == torch.float32
    rank = x.dim()
    dim = _normalize_dim(chunk_dim, rank)
    if dim == rank - 1:
        raise ValueError("chunked low-memory modulation must not split the hidden normalization dimension.")
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive.")

    target_size = x.shape[dim]
    dtype = x.dtype
    out = torch.empty_like(x)
    for start in range(0, target_size, max_chunk_tokens):
        length = min(max_chunk_tokens, target_size - start)
        x_slice = x.narrow(dim, start, length)
        shift_slice = _modulation_param_slice(shift, dim=dim, start=start, length=length, target_size=target_size)
        scale_slice = _modulation_param_slice(scale, dim=dim, start=start, length=length, target_size=target_size)
        shift_slice = modulation_param_for_activation(shift_slice, x_slice)
        scale_slice = modulation_param_for_activation(scale_slice, x_slice)
        chunk = _low_memory_norm(norm_func, x_slice).to(dtype=dtype)
        chunk = chunk * (scale_slice + 1) + shift_slice
        out.narrow(dim, start, length).copy_(chunk)
    return out


def modulation_param_for_activation(value, activation):
    if _device_type(activation) == "mps" and activation.dtype in (torch.bfloat16, torch.float16):
        return value.to(dtype=activation.dtype)
    return value


def _low_memory_norm(norm_func, x):
    if isinstance(norm_func, nn.LayerNorm):
        weight = None if norm_func.weight is None else norm_func.weight.to(device=x.device, dtype=x.dtype)
        bias = None if norm_func.bias is None else norm_func.bias.to(device=x.device, dtype=x.dtype)
        return F.layer_norm(x, norm_func.normalized_shape, weight, bias, norm_func.eps)
    return norm_func(x).to(dtype=x.dtype)


def should_chunk_modulation(value, *, chunk_dim=-2, max_chunk_tokens=MPS_MODULATION_MAX_CHUNK_TOKENS):
    if _device_type(value) != "mps":
        return False
    try:
        dim = _normalize_dim(chunk_dim, value.dim())
    except (AttributeError, ValueError):
        return False
    return dim != value.dim() - 1 and value.shape[dim] > max_chunk_tokens


def modulate_fp32_memory_safe(
    norm_func,
    x,
    shift,
    scale,
    *,
    chunk_dim=-2,
    max_chunk_tokens=MPS_MODULATION_MAX_CHUNK_TOKENS,
):
    if should_chunk_modulation(x, chunk_dim=chunk_dim, max_chunk_tokens=max_chunk_tokens):
        if x.dtype in (torch.bfloat16, torch.float16):
            return modulate_mps_low_memory_chunked(
                norm_func,
                x,
                shift,
                scale,
                chunk_dim=chunk_dim,
                max_chunk_tokens=max_chunk_tokens,
            )
        return modulate_fp32_chunked(
            norm_func,
            x,
            shift,
            scale,
            chunk_dim=chunk_dim,
            max_chunk_tokens=max_chunk_tokens,
        )
    return modulate_fp32(norm_func, x, shift, scale)


def _device_type(value):
    device = getattr(value, "device", value)
    device_type = getattr(device, "type", None)
    if device_type is not None:
        return str(device_type)
    return str(device).split(":", 1)[0]


def fp32_modulation_context(value):
    if _device_type(value) == "cuda":
        return amp.autocast(device_type="cuda", dtype=torch.float32)
    return nullcontext()


class FinalLayer_FP32(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, num_patch, out_channels, adaln_tembed_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patch = num_patch
        self.out_channels = out_channels
        self.adaln_tembed_dim = adaln_tembed_dim

        self.norm_final = LayerNorm_FP32(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 2 * hidden_size, bias=True))

    def forward(self, x, t, latent_shape):
        # timestep shape: [B, T, C]
        assert t.dtype == torch.float32
        B, N, C = x.shape
        T, _, _ = latent_shape

        with fp32_modulation_context(x):
            shift, scale = self.adaLN_modulation(t).unsqueeze(2).chunk(2, dim=-1) # [B, T, 1, C]
            x = modulate_fp32(self.norm_final, x.view(B, T, -1, C), shift, scale).view(B, N, C)
            x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, t_embed_dim, frequency_embedding_size=256):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, t_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """

    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.y_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, caption):
        B, _, N, C = caption.shape
        caption = self.y_proj(caption)
        return caption
