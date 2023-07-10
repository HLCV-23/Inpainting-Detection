import unittest

import torch

from pyramid_vit.model import PyramidVisionTransformer

class TestPyramidVisionTransformer(unittest.TestCase):
    def test_output_size(self):
        input_dims = (3, 512, 512)
        out_channels = [64, 128, 320, 512]
        patch_sizes = [4, 2, 2, 2]
        encoders_per_stage = [2, 2, 2, 2]
        reduction_ratio = [8, 4, 2, 1]
        num_heads = [1, 2, 5, 8]
        mlp_ratio = [8, 8, 4, 4]
        model = PyramidVisionTransformer(
            input_dims=input_dims,
            out_channels=out_channels,
            patch_sizes=patch_sizes,
            encoders_per_stage=encoders_per_stage,
            reduction_ratio=reduction_ratio,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        input_tensor = torch.randn(1, *input_dims)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, (1, 512, 16, 16))





