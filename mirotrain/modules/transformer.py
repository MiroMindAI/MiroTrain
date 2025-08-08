# SPDX-FileCopyrightText: 2025 MiromindAI
# SPDX-FileCopyrightText: Meta Platforms, Inc. and affiliates

# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

# Adapted from torchtune.modules.transformer

from typing import Callable, Dict, List, Optional, Union

import torch

from mirotrain.modules.moe.utils import load_balancing_loss_func
from torch import nn
from torchtune.modules import (
    MultiHeadAttention,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)
from torchtune.modules.attention_utils import _MaskType


class SDTransformerSelfAttentionLayer(TransformerSelfAttentionLayer):
    """
    Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm (Optional[nn.Module]): Normalization to be applied before self-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        sa_scale (Optional[nn.Module]): Module to scale self-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.
        mask_mod (Optional[Callable[[_MaskType, int, int, int], _MaskType]]): A callable
            taking a _MaskType, bsz, and seq_len, and modifying the mask (e.g. for chunked attention).
    """

    def __init__(
        self,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        *,
        sa_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        sa_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
        mask_mod: Optional[Callable[[_MaskType, int, int, int], _MaskType]] = None,
    ) -> None:
        super().__init__(
            attn,
            mlp,
            sa_norm=sa_norm,
            mlp_norm=mlp_norm,
            sa_scale=sa_scale,
            mlp_scale=mlp_scale,
            mask_mod=mask_mod,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
            **kwargs (Dict): transformer layer inputs not relevant to self attention.

        Returns:
            torch.Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        h = self.sa_norm(x)
        if self.mask_mod is not None:
            # With TP we need to use a replicated tensor here
            bsz, seq_len, *_ = h.shape
            mask = self.mask_mod(mask=mask, bsz=bsz, seq_len=seq_len)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos, **kwargs)
        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.sa_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out


class SDTransformerDecoder(TransformerDecoder):
    """
    Transformer Decoder derived from the Llama2 architecture.

    Args:
        tok_embeddings (nn.Embedding): PyTorch embedding layer, to be used to move
            tokens to an embedding space.
        layers (Union[nn.Module, List[nn.Module], nn.ModuleList]): A single transformer Decoder layer, an
            nn.ModuleList of layers or a list of layers. It is recommended to use an nn.ModuleList.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value. This is used to setup the
            :func:`~torchtune.modules.KVCache`
        head_dim (int): embedding dimension for each head in self-attention. This is used
            to setup the :func:`~torchtune.modules.KVCache`
        norm (nn.Module): Callable that applies normalization to the output of the decoder,
            before final MLP.
        output (Union[nn.Linear, Callable]): Callable that applies a linear transformation to the output of
            the decoder.
        num_layers (Optional[int]): Number of Transformer Decoder layers, only define when
            layers is not a list.
        output_hidden_states (Optional[List[int]]): List of layers (indices) to include in the output

    Note:
        Arg values are checked for correctness (eg: ``attn_dropout`` belongs to [0,1])
        in the module where they are used. This helps reduces the number of raise
        statements in code and improves readability.
    """

    def __init__(
        self,
        *,
        tok_embeddings: nn.Embedding,
        layers: Union[nn.Module, List[nn.Module], nn.ModuleList],
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        output: Union[nn.Linear, Callable],
        num_layers: Optional[int] = None,
        output_hidden_states: Optional[List[int]] = None,
    ) -> None:
        super().__init__(
            tok_embeddings=tok_embeddings,
            layers=layers,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            norm=norm,
            output=output,
            num_layers=num_layers,
            output_hidden_states=output_hidden_states,
        )
        self.enable_cpu_offload_for_chunks = False

    def set_cpu_offload_for_chunks(self, enable: bool) -> None:
        """Enable or disable CPU offload for chunked output to save GPU memory."""
        self.enable_cpu_offload_for_chunks = enable

    def chunked_output_with_cpu_offload(
        self, last_hidden_state: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Apply output projection in chunks with immediate CPU offload for memory efficiency.
        Each chunk is computed and immediately moved to CPU to reduce GPU memory spikes.

        Args:
            last_hidden_state (torch.Tensor): last hidden state of the decoder, having shape
                [b, seq_len, embed_dim].

        Returns:
            List[torch.Tensor]: List of num_chunks output tensors on CPU, each with shape
                [b, seq_len/num_chunks, out_dim], where out_dim is usually the vocab size.

        Raises:
            ValueError: If ``num_output_chunks`` is not set (<= 0) before using this method.
        """
        if self.num_output_chunks <= 0:
            raise ValueError(
                "num_output_chunks must be set before using chunked_output_with_cpu_offload"
            )

        chunks = last_hidden_state.tensor_split(self.num_output_chunks, dim=1)
        output_chunks = []

        for chunk in chunks:
            chunk_logits = self.output(chunk)

            # Immediately move to CPU if enabled
            if self.enable_cpu_offload_for_chunks:
                chunk_logits = chunk_logits.cpu()

            output_chunks.append(chunk_logits)

            # Clear GPU memory for this chunk
            del chunk

        return output_chunks

    def forward(
        self,
        tokens: Optional[torch.Tensor],
        *,
        mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            tokens (Optional[torch.Tensor]): input tensor with shape ``[b x s]``
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. This parameter is required during inference if caches have been setup.
                Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape ``[b x s_e x d_e]``
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position ``i,j`` means token ``i`` can attend
                to embedding ``j`` in the decoder. Mask has shape ``[b x s x s_e]``. Default is None,
                but this is required during inference if the model has been setup with any layers
                which use encoder embeddings and caches have been setup.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current token.
                This parameter is required during inference if caches have been setup. Default is None.
            input_embeds (Optional[torch.Tensor]): Pass these instead of tokens to short-circuit token embeddings
                and skip straight to the transformer layers. Shape ``[b x s x d]``. Default: None
            **kwargs (Dict): Optional additional arguments.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: output tensor with shape ``[b x s x v]`` if `self.skip_output_layer=False`
            and ``[b x s x d]`` otherwise, or a list of layer output tensors defined by ``output_hidden_states`` with the
            final output tensor appended to the list.

        Note:
            At the very first step of inference, when the model is provided with a prompt,
            ``input_pos`` should contain the positions of all of the tokens in the prompt.
            For a single-batch prompt, or a batch of prompts with identical lengths, this
            will be ``torch.arange(prompt_length)``. For a batch of varying-length prompts,
            shorter prompts are left-padded and position ids are correspondingly right-shifted,
            thus positional ids should be of shape ``[b, padded_prompt_length]``.
            This is because we will need to retrieve the positional embeddings for each input id.
            In the subsequent steps, if the model has been setup with KV-caches, ``input_pos`` will contain
            the position(s) of the current token(s) ``torch.tensor([padded_prompt_length])``. Otherwise,
            ``input_pos`` will contain all the position ids up to the current token.

        Shape notation:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """

        self._validate_inputs(
            tokens=tokens,
            mask=mask,
            encoder_input=encoder_input,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
            input_embeds=input_embeds,
        )

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens) if input_embeds is None else input_embeds

        hidden = []
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)
            # shape: [b, s, d]
            h = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
                **kwargs,
            )

        if len(self.layers) in self.output_hidden_states:
            hidden.append(h)

        # shape: [b, seq_len, out_dim]
        output = self.unembed(h)

        # Output list if hidden states are requested, otherwise just the output
        # TODO: always output a list to have a consistent output type
        output = output if not hidden else [*hidden, output]
        return output

    def unembed(self, h):
        # shape: [b, s, d]
        h = self.norm(h)
        if self.skip_output_layer:
            output = h
        elif self.num_output_chunks > 0:
            if self.enable_cpu_offload_for_chunks:
                output = self.chunked_output_with_cpu_offload(h)
            else:
                output = self.chunked_output(h)
        else:
            # shape: [b, seq_len, out_dim]
            output = self.output(h).float()

        return output


class MoETransformerSelfAttentionLayer(TransformerSelfAttentionLayer):
    """
    Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm (Optional[nn.Module]): Normalization to be applied before self-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        sa_scale (Optional[nn.Module]): Module to scale self-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.
        mask_mod (Optional[Callable[[_MaskType, int, int, int], _MaskType]]): A callable
            taking a _MaskType, bsz, and seq_len, and modifying the mask (e.g. for chunked attention).
    """

    def __init__(
        self,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        *,
        sa_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        sa_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
        mask_mod: Optional[Callable[[_MaskType, int, int, int], _MaskType]] = None,
    ) -> None:
        super().__init__(
            attn=attn,
            mlp=mlp,
            sa_norm=sa_norm,
            mlp_norm=mlp_norm,
            sa_scale=sa_scale,
            mlp_scale=mlp_scale,
            mask_mod=mask_mod,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
            **kwargs (Dict): transformer layer inputs not relevant to self attention.

        Returns:
            torch.Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        h = self.sa_norm(x)
        if self.mask_mod is not None:
            # With TP we need to use a replicated tensor here
            bsz, seq_len, *_ = h.shape
            mask = self.mask_mod(mask=mask, bsz=bsz, seq_len=seq_len)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos, **kwargs)
        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.sa_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))

        router_logits = None
        if isinstance(mlp_out, tuple):
            router_logits = mlp_out[1]
            mlp_out = mlp_out[0]

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)

        if router_logits is not None:
            return out, router_logits
        return out


class MoETransformerDecoder(SDTransformerDecoder):
    """
    Transformer Decoder derived from the Llama2 architecture.

    Args:
        tok_embeddings (nn.Embedding): PyTorch embedding layer, to be used to move
            tokens to an embedding space.
        layers (Union[nn.Module, List[nn.Module], nn.ModuleList]): A single transformer Decoder layer, an
            nn.ModuleList of layers or a list of layers. It is recommended to use an nn.ModuleList.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value. This is used to setup the
            :func:`~torchtune.modules.KVCache`
        head_dim (int): embedding dimension for each head in self-attention. This is used
            to setup the :func:`~torchtune.modules.KVCache`
        norm (nn.Module): Callable that applies normalization to the output of the decoder,
            before final MLP.
        output (Union[nn.Linear, Callable]): Callable that applies a linear transformation to the output of
            the decoder.
        num_layers (Optional[int]): Number of Transformer Decoder layers, only define when
            layers is not a list.
        output_hidden_states (Optional[List[int]]): List of layers (indices) to include in the output
        output_router_logits (bool): Whether to return moe layer gate's logits output.
        num_experts (Optional[int]): Number of experts in the moe layer.
        top_k (Optional[int]): Top k experts to select for load balancing.

    Note:
        Arg values are checked for correctness (eg: ``attn_dropout`` belongs to [0,1])
        in the module where they are used. This helps reduces the number of raise
        statements in code and improves readability.
    """

    def __init__(
        self,
        *,
        tok_embeddings: nn.Embedding,
        layers: Union[nn.Module, List[nn.Module], nn.ModuleList],
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        output: Union[nn.Linear, Callable],
        num_layers: Optional[int] = None,
        output_hidden_states: Optional[List[int]] = None,
        output_router_logits: bool = False,
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__(
            tok_embeddings=tok_embeddings,
            layers=layers,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            norm=norm,
            output=output,
            num_layers=num_layers,
            output_hidden_states=output_hidden_states,
        )

        # for moe router load balancing
        self.output_router_logits = output_router_logits
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        tokens: Optional[torch.Tensor],
        *,
        mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            tokens (Optional[torch.Tensor]): input tensor with shape ``[b x s]``
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. This parameter is required during inference if caches have been setup.
                Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape ``[b x s_e x d_e]``
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position ``i,j`` means token ``i`` can attend
                to embedding ``j`` in the decoder. Mask has shape ``[b x s x s_e]``. Default is None,
                but this is required during inference if the model has been setup with any layers
                which use encoder embeddings and caches have been setup.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current token.
                This parameter is required during inference if caches have been setup. Default is None.
            input_embeds (Optional[torch.Tensor]): Pass these instead of tokens to short-circuit token embeddings
                and skip straight to the transformer layers. Shape ``[b x s x d]``. Default: None
            attention_mask (Optional[torch.Tensor]): Attention mask for the input tokens.
                Shape ``[b x s]``. Default: None
            **kwargs (Dict): Optional additional arguments.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: output tensor with shape ``[b x s x v]`` if `self.skip_output_layer=False`
            and ``[b x s x d]`` otherwise, or a list of layer output tensors defined by ``output_hidden_states`` with the
            final output tensor appended to the list.

        Note:
            At the very first step of inference, when the model is provided with a prompt,
            ``input_pos`` should contain the positions of all of the tokens in the prompt.
            For a single-batch prompt, or a batch of prompts with identical lengths, this
            will be ``torch.arange(prompt_length)``. For a batch of varying-length prompts,
            shorter prompts are left-padded and position ids are correspondingly right-shifted,
            thus positional ids should be of shape ``[b, padded_prompt_length]``.
            This is because we will need to retrieve the positional embeddings for each input id.
            In the subsequent steps, if the model has been setup with KV-caches, ``input_pos`` will contain
            the position(s) of the current token(s) ``torch.tensor([padded_prompt_length])``. Otherwise,
            ``input_pos`` will contain all the position ids up to the current token.

        Shape notation:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """

        self._validate_inputs(
            tokens=tokens,
            mask=mask,
            encoder_input=encoder_input,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
            input_embeds=input_embeds,
        )

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens) if input_embeds is None else input_embeds

        hidden = []
        all_router_logits = () if self.output_router_logits else None
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)
            # shape: [b, s, d]
            h = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
                **kwargs,
            )
            if self.output_router_logits:
                all_router_logits += (h[-1],)
            if isinstance(h, tuple):
                h = h[0]

        if len(self.layers) in self.output_hidden_states:
            hidden.append(h)

        # shape: [b, seq_len, out_dim]
        output = self.unembed(h)

        # Output list if hidden states are requested, otherwise just the output
        # TODO: always output a list to have a consistent output type
        output = output if not hidden else [*hidden, output]

        # support aux-loss for moe router load balancing
        load_balancing_loss = None
        if self.output_router_logits:
            assert (
                attention_mask is not None
            ), "attention_mask is required for computing load balancing loss"
            load_balancing_loss = load_balancing_loss_func(
                gate_logits=all_router_logits,
                num_experts=self.num_experts,
                top_k=self.top_k,
                attention_mask=attention_mask,
            )

        return output, load_balancing_loss
