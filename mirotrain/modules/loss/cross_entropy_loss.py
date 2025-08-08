# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from torch import nn
from torch.distributed.tensor import DTensor

from torchtune.modules.loss.loss_types import SFTLoss
from torchtune.utils import get_logger

log = get_logger()


class LigerLinearCrossEntropyLoss(nn.Module, SFTLoss):
    """
    Memory-efficient cross-entropy loss using Liger's fused implementation.

    This loss function integrates the final linear projection with the cross-entropy calculation,
    reducing memory overhead and increasing performance by avoiding intermediate activations.

    This requires the model to skip the final output projection, which will be handled by this class.

    Usage:
        >>> model = Transformer(...)
        >>> loss = LigerLinearCrossEntropyLoss(...)
        >>> loss.set_model_output(model)
        >>> loss.apply_compile_strategy()
    """

    def __init__(
        self,
        ignore_index: int = -100,
    ):
        # ignore_index (int): Token index to ignore when computing loss (typically padding tokens).
        super().__init__()
        self.linear_projection = None
        self.ignore_index = ignore_index
        self.loss_fn = LigerFusedLinearCrossEntropyLoss(
            ignore_index=self.ignore_index, reduction="mean"
        )

    def set_model_output(self, model: nn.Module) -> None:
        """
        Registers the model's output projection layer to be used inside the loss function.

        Args:
            model (nn.Module): The model whose output layer will be used for projection.
        """
        model.skip_output_layer = True
        self.linear_projection = model.output

    def apply_compile_strategy(self, *args, **kwargs):
        """
        Stub for applying compilation. Currently not supported for this fused loss.
        """
        log.warning("Skipping compile loss, as it is not supported at this time")
        return self

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the Liger fused linear + cross-entropy loss.

        Args:
            outputs (torch.Tensor): Model hidden states before projection. Shape: [batch_size, seq_len, embed_dim]
            targets (torch.Tensor): Ground truth token indices. Shape: [batch_size, seq_len]

        Returns:
            torch.Tensor: Scalar loss tensor.
        """

        # Flatten outputs and targets to [batch_size * seq_len, ...]
        targets = targets.reshape(-1)
        outputs = outputs.reshape(-1, outputs.size(-1))

        if (
            hasattr(self.linear_projection, "tied_module")
            and self.linear_projection.tied_module is not None
        ):
            linear_weight = self.linear_projection.tied_module.weight
        else:
            linear_weight = self.linear_projection.weight
        # Compute fused projection + cross-entropy loss
        loss = self.loss_fn(linear_weight, outputs, targets)

        return loss


class LinearSqrtCrossEntropyLoss(nn.Module, SFTLoss):
    """Memory efficient Cross-entropy loss that applies square root token weighting.

    This loss function weights tokens by 1/sqrt(sequence_length) to give more importance
    to shorter sequences during training. This is useful for multimodal training where
    sequences may have varying lengths.

    Linear cross entropy masks out ignored tokens before the projection layer to save memory.
    You therefore need to skip the final projection layer in your model and pass it to the loss instead.
    You can setup the loss with the model and compile it as shown below.

    >>> model = Transformer(...)
    >>> loss = LinearSqrtCrossEntropyLoss(...)
    >>> loss.set_model_output(model)
    >>> loss.apply_compile_strategy()
    """

    def __init__(
        self,
        num_output_chunks: int = 8,
        ignore_index: int = -100,
    ):
        super().__init__()
        """
        Args:
            num_output_chunks (int): Number of chunks to split the output tensor into. Default is 8.
            ignore_index (int): Index to ignore in the target tensor. Default is -100.
        """
        self.linear_projection = None
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the compute_cross_entropy function.
        If compiling CE + chunking operation together, memory requirement is higher."""
        log.warning("Skipping compile loss, as it is not supported at this time")
        # TODO fix compile and re-enable
        # self.compute_cross_entropy = torch.compile(
        #     self.compute_cross_entropy, *args, **kwargs
        # )
        return self

    def set_model_output(self, model: nn.Module) -> None:
        """Modify model output to match the expected input for the loss function."""
        model.skip_output_layer = True
        self.linear_projection = model.output

    def _compute_token_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute token weights using square root weighting strategy.

        Args:
            targets: Target token ids [batch_size, seq_len]

        Returns:
            Token weights [batch_size, seq_len]
        """
        batch_size, seq_len = targets.shape

        # Create mask for valid tokens (not padding/ignore)
        valid_mask = targets != self.ignore_index

        # Count valid tokens per sequence
        l_values = valid_mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]

        # w_i = 1/l^0.5 for square root averaging
        weights = torch.ones_like(targets, dtype=torch.float) / torch.sqrt(
            l_values + 1e-8
        )

        # Zero out weights for ignored tokens
        weights = weights * valid_mask.float()

        return weights

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            outputs (torch.Tensor): Hidden state of the model, pre projection. Shape ``[bsz, seq_len, emb_dim]``
            targets (torch.Tensor): Labels for the model. Shape ``[bsz, seq_len]``

        Returns:
            torch.Tensor: loss tensor
        """

        # Compute token weights for square root weighting
        token_weights = self._compute_token_weights(targets)

        # Chunk along sequence dimension
        hidden_chunks = outputs.tensor_split(self.num_output_chunks, dim=1)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=1)
        weight_chunks = token_weights.tensor_split(self.num_output_chunks, dim=1)

        # Compute cross-entropy loss for the chunks with weighting
        total_weighted_loss = 0.0
        total_weights = 0.0

        for idx in range(len(hidden_chunks)):
            # Get the chunk data
            hidden_chunk = hidden_chunks[idx]
            target_chunk = target_chunks[idx]
            weight_chunk = weight_chunks[idx]

            # Compute individual token losses for this chunk
            # First, get the logits for this chunk
            if self.linear_projection is None:
                raise AttributeError("forward called before update_model")
            logits_chunk = self.linear_projection(
                hidden_chunk
            )  # [batch_size, chunk_size, vocab_size]
            if isinstance(logits_chunk, DTensor):
                logits_chunk = logits_chunk.full_tensor()

            # Flatten for cross entropy computation
            flat_logits = logits_chunk.view(-1, logits_chunk.size(-1))
            flat_targets = target_chunk.view(-1)

            # Compute individual token losses (no reduction)
            token_losses = F.cross_entropy(
                flat_logits.float(),
                flat_targets,
                reduction="none",
                ignore_index=self.ignore_index,
            )

            # Reshape back to [batch_size, chunk_size]
            token_losses = token_losses.view(target_chunk.shape)

            # Apply weights to individual token losses
            weighted_token_losses = token_losses * weight_chunk

            # Sum up weighted losses and weights for this chunk
            valid_mask_chunk = target_chunk != self.ignore_index
            chunk_weighted_loss = weighted_token_losses[valid_mask_chunk].sum()
            chunk_weights_sum = weight_chunk[valid_mask_chunk].sum()

            total_weighted_loss += chunk_weighted_loss
            total_weights += chunk_weights_sum

        if total_weights == 0:
            # must return after calling compute_cross_entropy to not hang during data parallel training
            return total_weighted_loss
        else:
            return total_weighted_loss / total_weights
