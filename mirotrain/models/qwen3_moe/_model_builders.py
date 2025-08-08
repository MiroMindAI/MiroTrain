# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtune.models.qwen2._model_builders

from mirotrain.modules.transformer import MoETransformerDecoder

from ._component_builders import qwen3_moe

"""
Model builders build specific instantiations using component builders. For example
the qwen3_30b_a3b model builder uses the qwen3_moe component builder to create the
Qwen3 30B MoE model.
"""


def qwen3_30b_a3b(router_aux_loss_coef: float = 0.001, use_grouped_gemm: bool = True) -> MoETransformerDecoder:
    """
    Builder for creating a Qwen3 MoE 30B base model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-30B-A3B

    Returns:
        TransformerDecoder: Instantiation of Qwen3 MoE 30B base model
    """
    return qwen3_moe(
        vocab_size=151936,
        num_layers=48,
        num_heads=32,
        num_kv_heads=4,
        embed_dim=2048,
        intermediate_dim=6144,
        moe_intermediate_dim=768,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=False,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        num_experts=128,
        num_experts_per_tok=8,
        norm_topk_prob=True,
        mlp_only_layers=[],
        moe_every_n_layers=1,
        output_router_logits=True,
        router_aux_loss_coef=router_aux_loss_coef,
        use_grouped_gemm=use_grouped_gemm,
    )


def qwen3_235b_a22b(router_aux_loss_coef: float = 0.001, use_grouped_gemm: bool = True) -> MoETransformerDecoder:
    """
    Builder for creating a Qwen3 MoE 235B base model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-235B-A22B

    Returns:
        TransformerDecoder: Instantiation of Qwen3 MoE 30B base model
    """
    return qwen3_moe(
        vocab_size=151936,
        num_layers=94,
        num_heads=64,
        num_kv_heads=4,
        embed_dim=4096,
        intermediate_dim=12288,
        moe_intermediate_dim=1536,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=False,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        num_experts=128,
        num_experts_per_tok=8,
        norm_topk_prob=True,
        mlp_only_layers=[],
        moe_every_n_layers=1,
        output_router_logits=True,
        router_aux_loss_coef=router_aux_loss_coef,
        use_grouped_gemm=use_grouped_gemm,
    )
