import datetime
import os
import pickle
import subprocess
import time
from collections import defaultdict, deque
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _gelu_ignore_parameters(*args, **kwargs) -> nn.Module:
    """Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.
    Args:
        *args: Ignored.
        **kwargs: Ignored.
    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation


def format_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"
