from functools import partial
from typing import Any

import torch
from timm.models.vision_transformer import _cfg
from torch import nn

from idisc.models.backbones import (Bottleneck, EfficientNet, ResNet,
                                    SwinTransformer, _resnet)


def swin_tiny(pretrained: bool = True, **kwargs):
    if pretrained:
        pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
    model = SwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dims=[96 * (2**i) for i in range(4)],
        num_heads=[3, 6, 12, 24],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 6, 2],
        drop_path_rate=0.2,
        pretrained=pretrained,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


def swin_small(pretrained: bool = True, **kwargs):
    if pretrained:
        pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth"
    model = SwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dims=[96 * (2**i) for i in range(4)],
        num_heads=[3, 6, 12, 24],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3,
        pretrained=pretrained,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


def swin_base(pretrained: bool = True, **kwargs):
    if pretrained:
        pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth"
    model = SwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dims=[128 * (2**i) for i in range(4)],
        num_heads=[4, 8, 16, 32],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 18, 2],
        drop_path_rate=0.5,
        pretrained=pretrained,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


def swin_large_22k(pretrained: bool = True, **kwargs):
    if pretrained:
        pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth"
    model = SwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dims=[192 * (2**i) for i in range(4)],
        num_heads=[6, 12, 24, 48],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 18, 2],
        drop_path_rate=0.2,
        pretrained=pretrained,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


def resnet50(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def efficientnet_b5(pretrained: bool = True, **kwargs):
    basemodel_name = "tf_efficientnet_b5_ap"
    basemodel = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=pretrained
    )
    basemodel.global_pool = nn.Identity()
    basemodel.classifier = nn.Identity()
    return EfficientNet(basemodel, [5, 6, 8, 15])  # 11->15
