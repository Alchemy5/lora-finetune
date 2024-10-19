# Copyright 2024 Alpha-VLLM Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import math
import torch
import torch.nn as nn

from diffusers.utils import logging
from diffusers.models.attention import LuminaFeedForward
from diffusers.models.attention_processor import Attention, LuminaAttnProcessor2_0
from diffusers.models.embeddings import (
    LuminaPatchEmbed,
)

from diffusers.models.normalization import (
    RMSNorm,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def replace_linear_with_lora_linear(model: nn.Module, rank: int):
    def replace_with_lora_linear(model: nn.Module):
        for name, layer in model.named_children():
            if isinstance(layer, nn.Linear):
                lora_layer = LoraLinear(
                    in_features=layer.in_features,
                    out_features=layer.out_features,
                    rank=rank,
                )
                setattr(model, name, lora_layer)
            else:
                replace_with_lora_linear(layer)
        return model

    return replace_with_lora_linear(model)


class LoraLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.WA = nn.Parameter(torch.zeros(in_features, rank), requires_grad=True)
        self.WB = nn.Parameter(torch.zeros(rank, out_features), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.scaling = 1 / rank

        nn.init.kaiming_uniform_(self.WA, a=math.sqrt(5))
        nn.init.zeros_(self.WB)

    def forward(self, x: torch.Tensor):
        W = self.WA @ self.WB * self.scaling
        x = x @ W + self.b
        return x


class LuminaT2IAdapter(nn.Module):
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: Optional[int] = 2,
        in_channels: Optional[int] = 4,
        hidden_size: Optional[int] = 2304,
        num_layers: Optional[int] = 32,
        num_attention_heads: Optional[int] = 32,
        num_kv_heads: Optional[int] = None,
        multiple_of: Optional[int] = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: Optional[float] = 1e-5,
        learn_sigma: Optional[bool] = True,
        qk_norm: Optional[bool] = True,
        scaling_factor: Optional[float] = 1.0,
        rank: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling_factor = scaling_factor
        self.rank = rank

        self.condition_img_in = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(nn.Conv2d(16, in_channels, 3, padding=1)),
        )

        self.patch_embedder = LuminaPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            bias=True,
        )

        self.layers = nn.ModuleList(
            [
                LuminaT2IAdapterBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    rank,
                )
                for _ in range(num_layers)
            ]
        )

        self.adapter_outs = nn.ModuleList(
            [
                zero_module(nn.Linear(hidden_size, hidden_size))
                for i in range(num_layers)
            ]
        )

        assert (
            hidden_size // num_attention_heads
        ) % 4 == 0, "2d rope needs head dim to be divisible by 4"

    def forward(
        self,
        condition_img: torch.Tensor,
        image_rotary_emb: torch.Tensor,
    ) -> torch.Tensor:

        # donwsample condition_img to latent resolution
        hidden_states = self.condition_img_in(condition_img)

        hidden_states, mask, img_size, image_rotary_emb = self.patch_embedder(
            hidden_states, image_rotary_emb
        )
        image_rotary_emb = image_rotary_emb.to(hidden_states.device)

        # extract features from adapter blocks
        features = []
        for layer, adapter_out in zip(self.layers, self.adapter_outs):
            hidden_states = layer(
                hidden_states=hidden_states,
                image_rotary_emb=image_rotary_emb,
                attention_mask=mask,
            )
            feature = adapter_out(hidden_states)
            features.append(feature)

        return features


class LuminaT2IAdapterBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        rank: int,
        norm_elementwise_affine: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.rank = rank

        # Self-attention
        attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            qk_norm="layer_norm_across_heads" if qk_norm else None,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=LuminaAttnProcessor2_0(),
        )
        self.attn1 = replace_linear_with_lora_linear(attn1, rank)

        feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.feed_forward = replace_linear_with_lora_linear(feed_forward, rank)

        self.norm = RMSNorm(
            dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine
        )

        self.ffn_norm = RMSNorm(
            dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
    ):
        residual = hidden_states
        norm_hidden_states = self.norm(hidden_states)

        self_attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_hidden_states,
            attention_mask=attention_mask,
            query_rotary_emb=image_rotary_emb,
            key_rotary_emb=image_rotary_emb,
        )

        hidden_states = residual + self_attn_output.flatten(-2)
        mlp_output = self.feed_forward(self.ffn_norm(hidden_states))
        hidden_states = hidden_states + mlp_output

        return hidden_states
