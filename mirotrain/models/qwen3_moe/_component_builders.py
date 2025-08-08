# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtune.models.qwen2._component_builders.qwen2

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchtune.models.qwen2 import Qwen2RotaryPositionalEmbeddings
from torchtune.models.qwen2._component_builders import qwen2_mlp
from torchtune.modules import (
    RMSNorm,
    TiedLinear,
)
from torchtune.modules.moe import (
    GroupedExperts,
    MoE,
    TokenChoiceTopKRouter,
)

from mirotrain.modules.attention import MultiHeadAttentionWithUlysses
from mirotrain.modules.moe import DroplessMoELayer
from mirotrain.modules.transformer import MoETransformerDecoder, MoETransformerSelfAttentionLayer

"""
Component builders for the Qwen3 MoE model.

torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``MultiHeadAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


def qwen3_moe(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    moe_intermediate_dim: int,
    max_seq_len: int,
    head_dim: Optional[int] = None,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    rope_base: float = 1_000_000.0,
    tie_word_embeddings: bool = False,
    q_proj_bias: bool = True,
    k_proj_bias: bool = True,
    v_proj_bias: bool = True,
    q_norm: bool = False,
    k_norm: bool = False,
    num_experts: int = 1,
    num_experts_per_tok: int = 1,
    norm_topk_prob: bool = False,
    mlp_only_layers: list = [],
    moe_every_n_layers: int = 1,
    output_router_logits: bool = False,
    router_aux_loss_coef: float = 0.001,
    use_grouped_gemm: bool = True,
) -> MoETransformerDecoder:
    """
    Build the decoder associated with the Qwen2 model. This includes:
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
        moe_intermediate_dim (Optional[int]): intermediate dimension for sparse MoE MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`
        head_dim (Optional[int]): Dimension of each attention head. If not
            specified, it defaults to `embed_dim // num_heads`. In GQA, `head_dim` is not necessarily equal to
            `embed_dim // num_heads`, so this parameter allows the caller to explicitly specify a custom value.
        norm_eps (float): epsilon in RMS norms.
        rope_base (float): the base period of the RoPE embeddings.
        tie_word_embeddings (bool): whether the model's input and output word embeddings should be tied.
        q_proj_bias (bool): whether to use bias in the query projection.
        k_proj_bias (bool): whether to use bias in the key projection.
        v_proj_bias (bool): whether to use bias in the value projection.
        q_norm (bool): whether to use normalization in the query projection.
        k_norm (bool): whether to use normalization in the key projection.
        num_experts (int): use qwen3 moe mlp when num_experts > 1.
        num_experts_per_tok (int): number of experts for per token.
        norm_topk_prob (bool): whether to normalize the top k experts' prob.
        mlp_only_layers (list[int]): the layer ids that with only mlp.
        moe_every_n_layers (Optional[int]): Frequency of inserting MoE layers in the decoder.
            If set, every nth layer will be an MoE layer. Default: MoE every layer
        output_router_logits (bool): whether to return the router's output logits.
        router_aux_loss_coef (float): the coefficient for the router aux loss.
        use_grouped_gemm (bool): whether the moe model's experts use grouped gemm or not. True is suggested.

    Returns:
        TransformerDecoder: Instantiation of Qwen3 MoE model.
    """
    head_dim = head_dim or embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads

    rope = Qwen2RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)

    layers = nn.ModuleList()
    for layer_idx in range(num_layers):
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

        if (layer_idx not in mlp_only_layers) and (
            num_experts > 0 and (layer_idx + 1) % moe_every_n_layers == 0
        ):
            if use_grouped_gemm:
                # GroupedMLP with GroupedGEMM
                mlp = DroplessMoELayer(
                    hidden_dim=embed_dim,
                    intermediate_dim=moe_intermediate_dim,
                    num_experts=num_experts,
                    top_k=num_experts_per_tok,
                    normalize_expert_weights=norm_topk_prob,
                    output_router_logits=output_router_logits,
                )
            else:
                # Sequential MLP
                mlp = Qwen3MoeSparseMoeBlock(
                    hidden_dim=embed_dim,
                    intermediate_dim=moe_intermediate_dim,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    norm_topk_prob=norm_topk_prob,
                    output_router_logits=output_router_logits,
                )

            # 2. GroupedMLP without GroupedGEMM
            # mlp = qwen3_moe_block(dim=embed_dim, hidden_dim=moe_intermediate_dim, num_experts=num_experts, experts_per_token=num_experts_per_tok)
        else:
            mlp = qwen2_mlp(dim=embed_dim, hidden_dim=intermediate_dim)

        layer = MoETransformerSelfAttentionLayer(
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

    model = MoETransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
        output_router_logits=output_router_logits,
        num_experts=num_experts,
        top_k=num_experts_per_tok,
    )
    return model


def qwen3_moe_block(
    dim: int,
    hidden_dim: int,
    num_experts: int = 8,
    experts_per_token: int = 1,
    use_shared_expert: bool = False,
) -> MoE:
    """
    Build the MoE layer associated with the Qwen3 model.

    Args:
        dim (int): Input dimension of experts.
        hidden_dim (int): Hidden dimension of experts.
        num_experts (int): Number of experts in each MoE layer. Default: 8
        experts_per_token (int): Number of experts each token will be routed to in Token Choice.
        use_shared_expert (bool): Whether to use a shared expert or not. Default: False

    Returns:
        MoE: Instantiation of MoE layer.
    """
    router = TokenChoiceTopKRouter(
        gate=nn.Linear(dim, num_experts, bias=False),
        dim=dim,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
    )
    experts = GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts)
    shared_expert = (
        qwen2_mlp(dim=dim, hidden_dim=hidden_dim) if use_shared_expert else None
    )
    return MoE(
        experts=experts,
        router=router,
        shared_expert=shared_expert,
    )


class Qwen3MoeSparseMoeBlock(nn.Module):
    """An inefficient implementation of the Experts layer using Sequential Experts.
    """
    def __init__(self, hidden_dim, intermediate_dim, num_experts, num_experts_per_tok, norm_topk_prob, output_router_logits):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits

        # gating
        self.gate = nn.Linear(hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [qwen2_mlp(dim=hidden_dim, hidden_dim=intermediate_dim) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        if self.output_router_logits:
            return final_hidden_states, router_logits

        return final_hidden_states
