"""
Author: Luigi Piccinelli
Licensed under the ECL-2.0 license (https://opensource.org/license/ecl-2-0/)
"""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class AttentionLayer(nn.Module):
    def __init__(
        self,
        sink_dim: int,
        hidden_dim: int,
        source_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        pre_norm: bool = True,
        norm_layer=nn.LayerNorm,
        sink_competition: bool = False,
        qkv_bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.pre_norm = pre_norm
        assert (
            hidden_dim % num_heads
        ) == 0, "hidden_dim and num_heads are not divisible"
        self.scale = (hidden_dim // num_heads) ** -0.5
        self.num_heads = num_heads

        self.norm = norm_layer(sink_dim, eps=eps)
        self.norm_context = (
            norm_layer(source_dim, eps=eps) if source_dim is not None else None
        )

        self.to_q = nn.Linear(sink_dim, hidden_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(
            sink_dim if source_dim is None else source_dim,
            hidden_dim * 2,
            bias=qkv_bias,
        )
        self.to_out = nn.Linear(
            hidden_dim, sink_dim if output_dim is None else output_dim
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.sink_competition = sink_competition

    def forward(
        self, sink: torch.Tensor, source: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.pre_norm:
            sink = self.norm(sink)
            if source is not None:
                source = self.norm_context(source)

        q = self.to_q(sink)
        source = source if source is not None else sink
        k, v = self.to_kv(source).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.num_heads),
            (q, k, v),
        )
        similarity_matrix = torch.einsum("bid, bjd -> bij", q, k) * self.scale

        if self.sink_competition:
            attn = F.softmax(similarity_matrix, dim=-2) + self.eps
            attn = attn / torch.sum(attn, dim=(-1,), keepdim=True)
        else:
            attn = F.softmax(similarity_matrix, dim=-1)

        attn = self.dropout(attn)

        out = torch.einsum("bij, bjd -> bid", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.num_heads)
        out = self.to_out(out)
        if not self.pre_norm:
            out = self.norm(out)

        return out
