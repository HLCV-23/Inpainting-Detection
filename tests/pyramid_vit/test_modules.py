import unittest
import torch

from pyramid_vit.modules import (
    DownsamplingBlock,
    PositionalEncoding,
    SpatialReductionAttention,
    MLP,
    TransformerEncoder,
    PyramidBlock
)


class TestDownsamplingBlock(unittest.TestCase):
    def test_output_size(self):
        height, width, in_channels, out_channels, patch_size = 64, 64, 3, 16, 2
        layer = DownsamplingBlock(out_channels, patch_size)
        input_tensor = torch.randn(1, in_channels, height, width)
        output_tensor = layer(input_tensor)
        expected_size = (
            1,
            height // patch_size * width // patch_size,
            out_channels,
        )  # expected output size
        self.assertEqual(output_tensor.size(), expected_size)


class TestPositionalEncoding(unittest.TestCase):
    def test_output_size(self):
        height, width, out_channels = 32, 32, 16
        in_channels = height * width
        layer = PositionalEncoding(in_channels, out_channels, 0.0)
        input_tensor = torch.randn(1, in_channels, out_channels)
        output_tensor = layer(input_tensor)
        expected_size = (1, in_channels, out_channels)
        self.assertEqual(output_tensor.size(), expected_size)


class TestSpatialReductionAttention(unittest.TestCase):
    def test_output_size(self):
        height, width, embed_dim = 32, 32, 16
        layer = SpatialReductionAttention(embed_dim, 1, 0.0, 4)
        input_tensor = torch.randn(1, height * width, embed_dim)
        output_tensor = layer(input_tensor)
        expected_size = (1, height * width, embed_dim)
        self.assertEqual(output_tensor.size(), expected_size)


class TestMLP(unittest.TestCase):
    def test_output_size(self):
        height, width, embed_dim = 32, 32, 16
        layer = MLP(embed_dim, 4)
        input_tensor = torch.randn(1, height * width, embed_dim)
        output_tensor = layer(input_tensor)
        expected_size = (1, height * width, embed_dim)
        self.assertEqual(output_tensor.size(), expected_size)


class TestTransformerBlock(unittest.TestCase):
    def test_output_size(self):
        embed_dim, num_heads, mlp_ratio, reduction_ratio = 16, 1, 4, 4
        height, width = 32, 32
        layer = TransformerEncoder(embed_dim, num_heads, mlp_ratio, reduction_ratio)
        input_tensor = torch.randn(1, height * width, embed_dim)
        output_tensor = layer(input_tensor)
        expected_size = (1, height * width, embed_dim)
        self.assertEqual(output_tensor.size(), expected_size)

class TestPyramidBlock(unittest.TestCase):

    def test_output_size(self):
        height, width, out_channels, patch_size = 64, 64, 16, 2
        num_encoders, reduction_ratio, num_heads, mlp_ratio = 1, 4, 1, 4
        layer = PyramidBlock(height, width, out_channels, patch_size, num_encoders, reduction_ratio, num_heads, mlp_ratio)
        input_tensor = torch.randn(1, 3, height, width)
        output_tensor = layer(input_tensor)
        expected_size = (1, out_channels, height // patch_size, width // patch_size)
        self.assertEqual(output_tensor.size(), expected_size)

