from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import SanaMultiscaleAttentionProjection, SanaMultiscaleAttnProcessor2_0
from diffusers.models.normalization import RMSNorm, get_normalization
from diffusers.models.transformers.sana_transformer import GLUMBConv


class SanaMultiscaleLinearAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, attention_head_dim: int = 32,
                 mult: float = 1.0, norm_type: str = 'batch_norm',
                 kernel_sizes: Tuple[int, ...] = (5,), eps: float = 1e-15,
                 residual_connection: bool = False):
        super().__init__()
        self.eps = eps
        self.attention_head_dim = attention_head_dim
        self.norm_type = norm_type
        self.residual_connection = residual_connection
        heads = int(in_channels // attention_head_dim * mult)
        inner = heads * attention_head_dim
        self.to_q = nn.Linear(in_channels, inner, bias=False)
        self.to_k = nn.Linear(in_channels, inner, bias=False)
        self.to_v = nn.Linear(in_channels, inner, bias=False)
        self.to_qkv_multiscale = nn.ModuleList([
            SanaMultiscaleAttentionProjection(inner, heads, k) for k in kernel_sizes
        ])
        self.nonlinearity = nn.ReLU()
        self.to_out = nn.Linear(inner * (1 + len(kernel_sizes)), out_channels, bias=False)
        self.norm_out = get_normalization(norm_type, num_features=out_channels)
        self.processor = SanaMultiscaleAttnProcessor2_0()

    def apply_linear_attention(self, query, key, value):
        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)
        scores = torch.matmul(value, key.transpose(-1, -2))
        hidden_states = torch.matmul(scores, query).float()
        return hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)

    def apply_quadratic_attention(self, query, key, value):
        scores = torch.matmul(key.transpose(-1, -2), query).float()
        scores = scores / (torch.sum(scores, dim=2, keepdim=True) + self.eps)
        return torch.matmul(value.float(), scores).to(value.dtype)

    def forward(self, hidden_states):
        return self.processor(self, hidden_states)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 norm_type: str = 'batch_norm', act_fn: str = 'relu6'):
        super().__init__()
        self.norm_type = norm_type
        self.nonlinearity = get_activation(act_fn) if act_fn is not None else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.norm = get_normalization(norm_type, out_channels)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.norm_type == 'rms_norm':
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states + residual


class EfficientViTBlock(nn.Module):
    def __init__(self, in_channels: int, attention_head_dim: int = 32,
                 qkv_multiscales: Tuple[int, ...] = (5,), norm_type: str = 'batch_norm'):
        super().__init__()
        self.attn = SanaMultiscaleLinearAttention(
            in_channels, in_channels, attention_head_dim=attention_head_dim,
            norm_type=norm_type, kernel_sizes=qkv_multiscales,
            residual_connection=True)
        self.conv_out = GLUMBConv(in_channels, in_channels, norm_type='rms_norm')

    def forward(self, x):
        return self.conv_out(self.attn(x))


def get_block(block_type, in_channels, out_channels, attention_head_dim,
              norm_type, act_fn, qkv_multiscales=()):
    if block_type == 'ResBlock':
        return ResBlock(in_channels, out_channels, norm_type, act_fn)
    if block_type == 'EfficientViTBlock':
        return EfficientViTBlock(in_channels, attention_head_dim, qkv_multiscales, norm_type)
    raise ValueError(f'Unsupported block type: {block_type}')


class DCUpBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, interpolate=False,
                 shortcut=True, interpolation_mode='nearest'):
        super().__init__()
        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = out_channels * self.factor ** 2 // in_channels
        if not interpolate:
            out_channels = out_channels * self.factor ** 2
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, hidden_states):
        if self.interpolate:
            x = F.interpolate(hidden_states, scale_factor=self.factor, mode=self.interpolation_mode)
            x = self.conv(x)
        else:
            x = F.pixel_shuffle(self.conv(hidden_states), self.factor)
        if self.shortcut:
            y = F.pixel_shuffle(hidden_states.repeat_interleave(self.repeats, dim=1), self.factor)
            x = x + y
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=16, attention_head_dim=32,
                 block_type=('ResBlock', 'ResBlock', 'ResBlock', 'EfficientViTBlock', 'EfficientViTBlock', 'EfficientViTBlock'),
                 block_out_channels=(64, 128, 256, 256, 512, 512),
                 layers_per_block=(0, 5, 10, 2, 2, 2),
                 qkv_multiscales=((), (), (), (), (), ()),
                 norm_type=('batch_norm', 'batch_norm', 'batch_norm', 'rms_norm', 'rms_norm', 'rms_norm'),
                 act_fn=('relu', 'relu', 'relu', 'silu', 'silu', 'silu'),
                 upsample_block_type='pixel_shuffle', in_shortcut=True):
        super().__init__()
        n = len(block_out_channels)
        block_type = (block_type,) * n if isinstance(block_type, str) else tuple(block_type)
        norm_type = (norm_type,) * n if isinstance(norm_type, str) else tuple(norm_type)
        act_fn = (act_fn,) * n if isinstance(act_fn, str) else tuple(act_fn)
        self.conv_in = nn.Conv2d(latent_channels, block_out_channels[-1], 3, 1, 1)
        self.in_shortcut = in_shortcut
        if in_shortcut:
            self.in_shortcut_repeats = block_out_channels[-1] // latent_channels
        up_blocks = []
        for i, (out_channel, num_layers) in reversed(list(enumerate(zip(block_out_channels, layers_per_block)))):
            items = []
            if i < n - 1 and num_layers > 0:
                items.append(DCUpBlock2d(
                    block_out_channels[i + 1], out_channel,
                    interpolate=upsample_block_type == 'interpolate', shortcut=True))
            for _ in range(num_layers):
                items.append(get_block(block_type[i], out_channel, out_channel,
                                       attention_head_dim, norm_type[i], act_fn[i], qkv_multiscales[i]))
            up_blocks.insert(0, nn.Sequential(*items))
        self.up_blocks = nn.ModuleList(up_blocks)
        channels = block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1]
        self.norm_out = RMSNorm(channels, 1e-5, elementwise_affine=True, bias=True)
        self.conv_act = nn.ReLU()
        self.conv_out = nn.Conv2d(channels, in_channels, 3, 1, 1) if layers_per_block[0] > 0 else DCUpBlock2d(
            channels, in_channels, interpolate=upsample_block_type == 'interpolate', shortcut=False)

    def forward(self, hidden_states):
        if self.in_shortcut:
            hidden_states = self.conv_in(hidden_states) + hidden_states.repeat_interleave(self.in_shortcut_repeats, dim=1)
        else:
            hidden_states = self.conv_in(hidden_states)
        for block in reversed(self.up_blocks):
            hidden_states = block(hidden_states)
        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        return self.conv_out(self.conv_act(hidden_states))
