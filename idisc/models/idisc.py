"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from idisc.models.defattn_decoder import MSDeformAttnPixelDecoder
from idisc.models.fpn_decoder import BasePixelDecoder
from idisc.models.id_module import AFP, ISD


class IDisc(nn.Module):
    def __init__(
        self,
        pixel_encoder: nn.Module,
        afp: nn.Module,
        pixel_decoder: nn.Module,
        isd: nn.Module,
        loss: nn.Module,
        afp_min_resolution=1,
        eps: float = 1e-6,
        **kwargs
    ):
        super().__init__()
        self.eps = eps
        self.pixel_encoder = pixel_encoder
        self.afp = afp
        self.pixel_decoder = pixel_decoder
        self.isd = isd
        self.afp_min_resolution = afp_min_resolution
        self.loss = loss

    def invert_encoder_output_order(
        self, xs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(xs[::-1])

    def filter_decoder_relevant_resolutions(
        self, decoder_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(decoder_outputs[self.afp_min_resolution :])

    def forward(
        self,
        image: torch.Tensor,
        gt: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        losses = {"opt": {}, "stat": {}}
        original_shape = gt.shape[-2:] if gt is not None else image.shape[-2:]

        encoder_outputs = self.pixel_encoder(image)
        encoder_outputs = self.invert_encoder_output_order(encoder_outputs)

        # DefAttn Decoder + filter useful resolutions (usually skip the lowest one)
        fpn_outputs, decoder_outputs = self.pixel_decoder(encoder_outputs)

        decoder_outputs = self.filter_decoder_relevant_resolutions(decoder_outputs)
        fpn_outputs = self.filter_decoder_relevant_resolutions(fpn_outputs)

        idrs = self.afp(decoder_outputs)
        outs = self.isd(fpn_outputs, idrs)

        out_lst = []
        for out in outs:
            if out.shape[1] == 1:
                out = F.interpolate(
                    torch.exp(out),
                    size=outs[-1].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                out = self.normalize_normals(
                    F.interpolate(
                        out,
                        size=outs[-1].shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                )
            out_lst.append(out)

        out = F.interpolate(
            torch.mean(torch.stack(out_lst, dim=0), dim=0),
            original_shape,
            # Legacy code for reproducibility for normals...
            mode="bilinear" if out.shape[1] == 1 else "bicubic",
            align_corners=True,
        )
        if gt is not None:
            losses["opt"] = {
                self.loss.name: self.loss.weight
                * self.loss(out, target=gt, mask=mask.bool(), interpolate=True)
            }
        return (
            out if out.shape[1] == 1 else out[:, :3],
            losses,
            {"outs": outs, "queries": idrs},
        )

    def normalize_normals(self, norms):
        min_kappa = 0.01
        norm_x, norm_y, norm_z, kappa = torch.split(norms, 1, dim=1)
        norm = torch.sqrt(norm_x**2.0 + norm_y**2.0 + norm_z**2.0 + 1e-6)
        kappa = F.elu(kappa) + 1.0 + min_kappa
        norms = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm, kappa], dim=1)
        return norms

    def load_pretrained(self, model_file):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        dict_model = torch.load(model_file, map_location=device)
        new_state_dict = deepcopy(
            {k.replace("module.", ""): v for k, v in dict_model.items()}
        )
        self.load_state_dict(new_state_dict)

    def get_params(self, config):
        backbone_lr = config["model"]["pixel_encoder"].get(
            "lr_dedicated", config["training"]["lr"] / 10
        )
        params = [
            {"params": self.pixel_decoder.parameters()},
            {"params": self.afp.parameters()},
            {"params": self.isd.parameters()},
            {"params": self.pixel_encoder.parameters()},
        ]
        max_lrs = [config["training"]["lr"]] * 3 + [backbone_lr]
        return params, max_lrs

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def build(cls, config: Dict[str, Dict[str, Any]]):
        pixel_encoder_img_size = config["model"]["pixel_encoder"]["img_size"]
        pixel_encoder_pretrained = config["model"]["pixel_encoder"].get(
            "pretrained", None
        )
        config_backone = {"img_size": np.array(pixel_encoder_img_size)}
        if pixel_encoder_pretrained is not None:
            config_backone["pretrained"] = pixel_encoder_pretrained
        import importlib

        mod = importlib.import_module("idisc.models.encoder")
        pixel_encoder_factory = getattr(mod, config["model"]["pixel_encoder"]["name"])
        pixel_encoder = pixel_encoder_factory(**config_backone)

        pixel_encoder_embed_dims = getattr(pixel_encoder, "embed_dims")
        config["model"]["pixel_encoder"]["embed_dims"] = pixel_encoder_embed_dims

        pixel_decoder = (
            MSDeformAttnPixelDecoder.build(config)
            if config["model"]["attn_dec"]
            else BasePixelDecoder.build(config)
        )
        afp = AFP.build(config)
        isd = ISD.build(config)

        mod = importlib.import_module("idisc.optimization.losses")
        loss = getattr(mod, config["training"]["loss"]["name"]).build(config)

        return deepcopy(
            cls(
                pixel_encoder=pixel_encoder,
                pixel_decoder=pixel_decoder,
                afp=afp,
                isd=isd,
                loss=loss,
                afp_min_resolution=len(pixel_encoder_embed_dims)
                - config["model"]["isd"]["num_resolutions"],
            )
        )
