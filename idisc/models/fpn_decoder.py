"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from torch import nn
from torch.nn import functional as F

from idisc.utils import Conv2d, c2_xavier_fill, get_norm


class BasePixelDecoder(nn.Module):
    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int,
        output_dim: int,
        norm: Union[str, Callable] = "BN",
        **kwargs,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.in_features = input_dims[::-1]
        use_bias = norm == ""
        for idx, in_channels in enumerate(self.in_features):
            if idx == 0:
                output_norm = get_norm(norm, output_dim)
                output_conv = Conv2d(
                    in_channels,
                    output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)
            else:
                lateral_norm = get_norm(norm, hidden_dim)
                output_norm = get_norm(norm, output_dim)

                lateral_conv = Conv2d(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                    bias=use_bias,
                    norm=lateral_norm,
                )
                output_conv = Conv2d(
                    hidden_dim,
                    output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                c2_xavier_fill(lateral_conv)
                c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

    def forward(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        fpn_output = []
        for idx, f in enumerate(self.in_features):
            x = features[idx]
            if idx == 0:
                y = getattr(self, f"layer_{idx+1}")(x)
            else:
                cur_fpn = getattr(self, f"adapter_{idx+1}")(x)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = getattr(self, f"layer_{idx+1}")(y)
            fpn_output.append(y)
        return fpn_output, fpn_output

    @classmethod
    def build(cls, config):
        obj = cls(
            input_dims=config["model"]["pixel_encoder"]["embed_dims"],
            hidden_dim=config["model"]["pixel_decoder"]["hidden_dim"],
            output_dim=config["model"]["pixel_decoder"]["hidden_dim"],
        )
        return obj
