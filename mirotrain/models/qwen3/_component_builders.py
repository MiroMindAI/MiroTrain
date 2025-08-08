# SPDX-FileCopyrightText: 2025 MiromindAI
# SPDX-FileCopyrightText: Meta Platforms, Inc. and affiliates

# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

from typing import Optional

from torch import nn
from torchtune.models.qwen2._component_builders import qwen2_mlp
from torchtune.models.qwen2._positional_embeddings import Qwen2RotaryPositionalEmbeddings

from torchtune.modules import (
    RMSNorm,
    TiedLinear,
)

from mirotrain.modules import (
    MultiHeadAttentionWithUlysses,
    SDTransformerSelfAttentionLayer,
    SDTransformerDecoder,
)

from ._positional_embeddings import Qwen3YarnRotaryPositionalEmbedding, RopeScaling


"""
Component builders for the Qwen3 model.
"""


def qwen3(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    head_dim: Optional[int] = None,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    rope_base: float = 1_000_000.0,
    rope_scaling: Optional[RopeScaling] = None,
    tie_word_embeddings: bool = False,
    q_proj_bias: bool = True,
    k_proj_bias: bool = True,
    v_proj_bias: bool = True,
    q_norm: bool = False,
    k_norm: bool = False,
) -> SDTransformerDecoder:
    """
    Build the decoder associated with the Qwen3 model. This includes:
    - Token embeddings
    - num_layers number of TransformerSelfAttentionLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`
        head_dim (Optional[int]): Dimension of each attention head. If not
            specified, it defaults to `embed_dim // num_heads`. In GQA, `head_dim` is not necessarily equal to
            `embed_dim // num_heads`, so this parameter allows the caller to explicitly specify a custom value.
        norm_eps (float): epsilon in RMS norms.
        rope_base (float): the base period of the RoPE embeddings.
        rope_scaling (Optional[RopeScaling]): the RoPE scaling configuration for extended context.
        tie_word_embeddings (bool): whether the model's input and output word embeddings should be tied.
        q_proj_bias (bool): whether to use bias in the query projection.
        k_proj_bias (bool): whether to use bias in the key projection.
        v_proj_bias (bool): whether to use bias in the value projection.
        q_norm (bool): whether to use normalization in the query projection.
        k_norm (bool): whether to use normalization in the key projection.

    Returns:
        TransformerDecoder: Instantiation of Qwen3 model.
    """
    head_dim = head_dim or embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads

    if rope_scaling is not None:
        rope = Qwen3YarnRotaryPositionalEmbedding(
            dim=head_dim,
            max_position_embeddings=max_seq_len,
            base=rope_base,
            rope_scaling=rope_scaling,
        )
    else:
        rope = Qwen2RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)

    layers = nn.ModuleList()
    for _ in range(num_layers):
        self_attn = MultiHeadAttentionWithUlysses(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=q_proj_bias),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=k_proj_bias),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=v_proj_bias),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            q_norm=RMSNorm(dim=head_dim, eps=norm_eps) if q_norm else None, # norm on head_dim
            k_norm=RMSNorm(dim=head_dim, eps=norm_eps) if k_norm else None,
            kv_cache=None,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        mlp = qwen2_mlp(dim=embed_dim, hidden_dim=intermediate_dim)
        layer = SDTransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    if tie_word_embeddings:
        output_proj = TiedLinear(tok_embeddings)
    else:
        output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    return SDTransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )
