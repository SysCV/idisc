from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from idisc.utils.misc import is_main_process


class SILog(nn.Module):
    def __init__(self, weight: float):
        super(SILog, self).__init__()
        self.name: str = "SILog"
        self.weight = weight
        self.eps: float = 1e-6

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        interpolate: bool = True,
    ) -> torch.Tensor:
        if interpolate:
            input = F.interpolate(
                input, target.shape[-2:], mode="bilinear", align_corners=True
            )
        if mask is not None:
            input = input[mask]
            target = target[mask]

        log_error = torch.log(input + self.eps) - torch.log(target + self.eps)
        mean_sq_log_error = torch.pow(torch.mean(log_error), 2.0)

        scale_inv = torch.var(log_error)
        Dg = scale_inv + 0.15 * mean_sq_log_error
        return torch.sqrt(Dg + self.eps)

    @classmethod
    def build(cls, config):
        return cls(weight=config["training"]["loss"]["weight"])


class AngularLoss(nn.Module):
    def __init__(self, weight: float):
        super(AngularLoss, self).__init__()
        self.name: str = "AngularLoss"
        self.weight = weight

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        interpolate: bool = True,
    ) -> torch.Tensor:
        if mask.ndim > 3:
            mask = mask.squeeze()
        if interpolate:
            input = F.interpolate(
                input, target.shape[-2:], mode="bilinear", align_corners=True
            )
        loss = torch.cosine_similarity(input[:, :3], target, dim=1)
        mask = (
            mask.float()
            * (loss.detach() < 0.999).float()
            * (loss.detach() > -0.999).float()
        ).to(torch.bool)
        loss = loss[mask]
        kappa = input[:, 3][mask]

        loss = (
            -torch.log(torch.square(kappa) + 1)
            + kappa * torch.acos(loss)
            + torch.log(1 + torch.exp(-kappa * torch.pi))
        )
        return loss.mean()

    @classmethod
    def build(cls, config):
        return cls(weight=config["training"]["loss"]["weight"])
