""""Implement the pyramid vision transformer as described in https://arxiv.org/abs/2102.12122.
"""
from pyramid_vit.modules import PyramidBlock
from torch import nn
from typing import Tuple, List
import math


class PyramidVisionTransformer(nn.Module):
    def __init__(
            self,
            input_dims: Tuple[int, int, int],  # = (3, 224, 224),
            out_channels: List[int],  # = [64, 128, 320, 512],
            patch_sizes: List[int],  # = [4, 2, 2, 2],
            encoders_per_stage: List[int],  # = [2, 2, 2, 2],
            reduction_ratio: List[int],  # = [8, 4, 2, 1],
            num_heads: List[int],  # = [1, 2, 5, 8],
            mlp_ratio: List[int]  # =[8, 8, 4, 4]
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        num_stages = len(out_channels)
        for i in range(num_stages):
            stage_height = input_dims[1] // math.prod(patch_sizes[:i])
            stage_width = input_dims[2] // math.prod(patch_sizes[:i])
            self.stages.append(
                PyramidBlock(
                    height=stage_height,
                    width=stage_width,
                    out_channels=out_channels[i],
                    patch_size=patch_sizes[i],
                    num_encoders=encoders_per_stage[i],
                    reduction_ratio=reduction_ratio[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio[i]
                )
            )

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x
