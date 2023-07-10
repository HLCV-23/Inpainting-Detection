"""
Implement modules for building a pyramid vision transformer as described in https://arxiv.org/abs/2102.12122.
Downsampling operation has been modified according to https://arxiv.org/abs/2106.13797.
"""
from torch import nn
import torch


class DownsamplingBlock(nn.Module):
    """Downsampling block."""

    def __init__(self, out_channels: int, patch_size: int):
        super(DownsamplingBlock, self).__init__()
        self.proj = nn.LazyConv2d(
            out_channels,
            kernel_size=2 * patch_size - 1,
            stride=patch_size,
            padding=patch_size - 1,
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor):
        # Shape: (B, C_out, H//S, W//S)
        x = self.proj(x)
        # Shape: (B, H//S*W//S, C_out)
        x = x.flatten(2).transpose(1, 2).contiguous()
        # Shape: (B, H//S*W//S, C_out)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, in_channels, out_channels))
        mask = torch.arange(in_channels, dtype=torch.float32).reshape(
            -1, 1
        ) / torch.pow(
            10000, torch.arange(0, out_channels, 2, dtype=torch.float32) / out_channels
        )
        self.P[:, :, 0::2] = torch.sin(mask)
        self.P[:, :, 1::2] = torch.cos(mask)

    def forward(self, x):
        x = x + self.P.to(x.device)
        return self.dropout(x)


class SpatialReductionAttention(nn.Module):
    """Spatial reduction attention."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, reduction_ratio=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.reduction_ratio = reduction_ratio
        self.W_S = nn.LazyLinear(embed_dim)
        # Layer norm for Q
        self.norm1 = nn.LayerNorm(embed_dim)
        # Layer norm for spatially reduced K and V
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x is a tensor of flattened patches.
        batches, height_width, channels = x.shape

        # Spatial reduction as described in the original Pyramid ViT paper.
        x_sr = x.view(
            batches,
            height_width // self.reduction_ratio**2,
            (self.reduction_ratio**2) * channels,
        ).contiguous()
        x_sr = self.W_S(x_sr)
        x_sr = self.norm1(x_sr)

        x = self.norm2(x)
        # Second output is the attention weights, which we don't need.
        x, _ = self.attention(x, x_sr, x_sr)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: int):
        super().__init__()
        num_hidden = embed_dim * mlp_ratio
        self.dense1 = nn.LazyLinear(num_hidden)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(0.5)
        self.dense2 = nn.LazyLinear(embed_dim)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(self.dense1(x)))))


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        reduction_ratio: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.sra = SpatialReductionAttention(
            embed_dim, num_heads, dropout, reduction_ratio
        )
        self.mlp = MLP(embed_dim, mlp_ratio)

    def forward(self, x):
        x = x + self.sra(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PyramidBlock(nn.Module):
    """Block containing one stage of the Pyramid Vision Transformer."""

    def __init__(
        self,
        height: int,
        width: int,
        out_channels: int,
        patch_size: int,
        num_encoders: int,
        reduction_ratio: int,
        num_heads: int,
        mlp_ratio: int,
    ):
        super().__init__()
        self.downsampling = DownsamplingBlock(out_channels, patch_size)
        self.height_new = height // patch_size
        self.width_new = width // patch_size
        self.size_patch_embedding = self.height_new * self.width_new
        self.positional_encoding = PositionalEncoding(
            self.size_patch_embedding, out_channels
        )
        self.encoders = nn.ModuleList(
            [
                TransformerEncoder(out_channels, num_heads, mlp_ratio, reduction_ratio)
                for _ in range(num_encoders)
            ]
        )

    def forward(self, x):
        batches = x.shape[0]
        x = self.downsampling(x)
        x = self.positional_encoding(x)
        for encoder in self.encoders:
            x = encoder(x)
        return (
            x.view(batches, self.height_new, self.width_new, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
