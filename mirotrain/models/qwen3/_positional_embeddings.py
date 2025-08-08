# SPDX-FileCopyrightText: 2025 MiromindAI
# SPDX-FileCopyrightText: Meta Platforms, Inc. and affiliates

# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

import math
from typing import Literal, Optional, TypedDict

import torch
from torch import nn


class RopeScaling(TypedDict):
    rope_type: Literal["yarn"]
    factor: float
    original_max_position_embeddings: int
    attention_factor: float | None
    mscale: float | None
    mscale_all_dim: float | None
    beta_fast: float | None
    beta_slow: float | None
    partial_rotary_factor: float | None


class Qwen3YarnRotaryPositionalEmbedding(nn.Module):
    """
    Yarn RoPE for Qwen3 model.
    """

    # Adapted from torchtune.models.qwen2._positional_embeddings.Qwen2RotaryPositionalEmbeddings.__init__
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: RopeScaling,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = rope_scaling
        self.rope_init()

    def rope_init(self):
        # ========== START OF CODE ADAPTED FROM HUGGINGFACE TRANSFORMERS ==========
        # https://github.com/huggingface/transformers/blob/1234683309cec4696ae99f8ff2686dafef7d8840/src/transformers/modeling_rope_utils.py#L226
        base = self.base
        partial_rotary_factor = self.rope_scaling.get("partial_rotary_factor", 1.0)
        head_dim = self.dim
        dim = int(head_dim * partial_rotary_factor)
        factor = self.rope_scaling["factor"]
        attention_factor = self.rope_scaling.get("attention_factor")
        mscale = self.rope_scaling.get("mscale")
        mscale_all_dim = self.rope_scaling.get("mscale_all_dim")

        # NOTE: DeekSeek-V3 (and potentially other models) modify `max_position_embeddings` and have a
        # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
        # values to compute the default attention scaling factor, instead of using `factor`.
        if "original_max_position_embeddings" in self.rope_scaling:
            original_max_position_embeddings = self.rope_scaling[
                "original_max_position_embeddings"
            ]
            factor = self.max_position_embeddings / original_max_position_embeddings
        else:
            original_max_position_embeddings = self.max_position_embeddings

        def get_mscale(scale, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        # Sets the attention factor as suggested in the paper
        if attention_factor is None:
            if mscale and mscale_all_dim:
                attention_factor = float(
                    get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
                )
            else:
                attention_factor = get_mscale(factor)

        # Optional config options
        # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
        beta_fast = self.rope_scaling.get("beta_fast") or 32
        beta_slow = self.rope_scaling.get("beta_slow") or 1

        # Compute the inverse frequencies
        def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
            """Inverse dimension formula to find the dimension based on the number of rotations"""
            return (
                dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
            ) / (2 * math.log(base))

        def find_correction_range(
            low_rot, high_rot, dim, base, max_position_embeddings
        ):
            """Find dimension range bounds based on rotations"""
            low = math.floor(
                find_correction_dim(low_rot, dim, base, max_position_embeddings)
            )
            high = math.ceil(
                find_correction_dim(high_rot, dim, base, max_position_embeddings)
            )
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(min, max, dim):
            if min == max:
                max += 0.001  # Prevent singularity

            linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func

        # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
        # to expand the possible context length. In other words, interpolation = apply scaling factor.
        pos_freqs = base ** (torch.arange(0, dim, 2).to(dtype=torch.float) / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)

        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_max_position_embeddings
        )

        # Get n-dimensional rotational scaling corrected for extrapolation
        inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(
            dtype=torch.float
        )
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation * inv_freq_extrapolation_factor
        )
        # ========== END OF CODE ADAPTED FROM HUGGINGFACE TRANSFORMERS ==========

        self.register_buffer("theta", inv_freq, persistent=False)
        self.build_rope_cache(self.max_position_embeddings)
        self.cache.mul_(attention_factor)

    # Copied from torchtune.models.qwen2._positional_embeddings.Qwen2RotaryPositionalEmbeddings.build_rope_cache
    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # We cache the cos and sin embeddings instead of the IDs. This helps
        # ensure we have correct behavior when training with bf16
        # Size: [max_seq_len, (dim * 2)]
        freqs = torch.cat([idx_theta, idx_theta], dim=-1)
        cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    # Copied from torchtune.models.qwen2._positional_embeddings.Qwen2RotaryPositionalEmbeddings.forward
    def forward(
        self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)
        head_dim = x.size(-1)

        # extract the values based on whether input_pos is set or not. When
        # input_pos is provided, we're in inference mode
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d * 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d * 2]
        rope_cache = rope_cache.view(-1, seq_len, 1, head_dim * 2)

        # [b, s, 1, h_d]
        cos = rope_cache[..., :head_dim].to(x.dtype)
        sin = rope_cache[..., head_dim:].to(x.dtype)

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)

        # cos: [b, s, 1, h_d]
        # x: [b, s, n_h, h_d]
        x_out = (x * cos) + (rotated * sin)
        return x_out.type_as(x)
