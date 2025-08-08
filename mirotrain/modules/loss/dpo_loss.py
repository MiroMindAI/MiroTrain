# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.rlhf._types import ChosenRejectedOutputs


@dataclasses.dataclass
class RunningMoments:
    """
    Calculates the running mean and standard deviation of a data stream.
    Uses torch.distributed for distributed training support.
    """

    mean: float = 0
    std: float = 1
    var: float = 1
    count: float = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        if torch.distributed.is_initialized():
            xs_mean, xs_var, xs_count = self._get_global_statistics(xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += (delta * xs_count / tot_count).item()
        new_var = tot_sum / tot_count
        self.std = (new_var * tot_count / (tot_count - 1)).float().sqrt().item()
        self.var = new_var.item()
        self.count = tot_count

        return (
            xs_mean.item(),
            (xs_var * xs_count / (xs_count - 1)).float().sqrt().item(),
        )

    @torch.no_grad()
    def _get_global_statistics(
        self, xs: torch.Tensor, mask=None, device="cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Computes element-wise mean and variance of the tensor across processes using torch.distributed.
        """
        xs = xs.to(xs.device)
        sum_and_count = torch.tensor(
            [xs.sum(), (xs.numel() if mask is None else mask.sum())], device=xs.device
        )
        torch.distributed.all_reduce(sum_and_count, op=torch.distributed.ReduceOp.SUM)
        global_sum, count = sum_and_count
        global_mean = global_sum / count

        sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
        torch.distributed.all_reduce(sum_var, op=torch.distributed.ReduceOp.SUM)
        global_var = sum_var / count

        return global_mean.to(device), global_var.to(device), count.item()


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss module: https://arxiv.org/abs/2305.18290

    This module supports multiple loss types including sigmoid, bco_pair, and sft losses.
    Based on the implementation in HF's TRL library.

    Args:
        beta (float): Temperature parameter for the DPO loss, typically in the range of 0.1 to 0.5. Default is 0.1.
        label_smoothing (float): Parameter encoding uncertainty about the labels. Default is 0.
        loss_config (dict): Dictionary mapping loss_type to loss_weight. Supported loss_types: "sigmoid", "bco_pair", "sft".
                      Example: {"sigmoid": 1.0, "bco_pair": 0.5, "sft": 0.1}

    Raises:
        ValueError: If an unsupported loss type is provided in the loss_config dictionary.
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_config: dict = None,
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_config = loss_config if loss_config is not None else {"sigmoid": 1.0}

        # Validate loss types
        supported_losses = {"sigmoid", "bco_pair", "sft"}
        for loss_type in self.loss_config.keys():
            if loss_type not in supported_losses:
                raise ValueError(
                    f"Unsupported loss type: {loss_type}. Supported types: {supported_losses}"
                )

        # Initialize running moments for bco_pair loss if needed
        if "bco_pair" in self.loss_config:
            self.running = RunningMoments()

    def forward(
        self,
        policy_inputs: ChosenRejectedOutputs,
        reference_inputs: ChosenRejectedOutputs,
        chosen_logits: Optional[torch.Tensor] = None,
        chosen_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Compute the combined DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_inputs (ChosenRejectedOutputs): Policy log-probs and logits required for the calculation.
            reference_inputs (ChosenRejectedOutputs): Reference log-probs and logits required for the calculation.
            chosen_logits (Optional[torch.Tensor]): Logits from the policy model for chosen responses (required for SFT).
            chosen_labels (Optional[torch.Tensor]): Labels for the chosen responses (required for SFT).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]: A tuple containing:
                - total_loss: The combined weighted loss.
                - chosen_rewards: Combined chosen rewards from all loss types.
                - rejected_rewards: Combined rejected rewards from all loss types.
                - loss_dict: Dictionary containing individual loss components.

        Raises:
            ValueError: If chosen_logits and chosen_labels are required for SFT loss but not provided.
        """
        loss_dict = {}
        device = policy_inputs.chosen_logps.device
        total_loss = 0

        # Initialize combined rewards
        batch_size = policy_inputs.chosen_logps.size(0)
        combined_chosen_rewards = torch.zeros(batch_size, device=device)
        combined_rejected_rewards = torch.zeros(batch_size, device=device)

        # Compute each loss type and accumulate
        for loss_type, loss_weight in self.loss_config.items():
            if loss_type == "sigmoid":
                loss_value, chosen_rewards, rejected_rewards = self._sigmoid_loss(
                    policy_inputs, reference_inputs
                )
                loss_dict["sigmoid"] = {
                    "loss": loss_value,
                    "chosen_rewards": chosen_rewards,
                    "rejected_rewards": rejected_rewards,
                }
                total_loss += loss_weight * loss_value
                combined_chosen_rewards += loss_weight * chosen_rewards
                combined_rejected_rewards += loss_weight * rejected_rewards

            elif loss_type == "bco_pair":
                loss_value, chosen_rewards, rejected_rewards = self._bco_pair_loss(
                    policy_inputs, reference_inputs
                )
                loss_dict["bco_pair"] = {
                    "loss": loss_value,
                    "chosen_rewards": chosen_rewards,
                    "rejected_rewards": rejected_rewards,
                }
                total_loss += loss_weight * loss_value
                combined_chosen_rewards += loss_weight * chosen_rewards
                combined_rejected_rewards += loss_weight * rejected_rewards

            elif loss_type == "sft":
                if chosen_logits is None or chosen_labels is None:
                    raise ValueError(
                        "chosen_logits and chosen_labels are required for SFT loss"
                    )

                loss_value, chosen_rewards, rejected_rewards = self._sft_loss(
                    chosen_logits, chosen_labels
                )
                loss_dict["sft"] = {
                    "loss": loss_value,
                    "chosen_rewards": chosen_rewards,
                    "rejected_rewards": rejected_rewards,
                }
                total_loss += loss_weight * loss_value
                # SFT rewards are zeros, so no need to add them

        return total_loss, combined_chosen_rewards, combined_rejected_rewards, loss_dict

    def _sigmoid_loss(
        self,
        policy_inputs: ChosenRejectedOutputs,
        reference_inputs: ChosenRejectedOutputs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the standard sigmoid-based DPO loss."""
        pi_logratios = policy_inputs.chosen_logps - policy_inputs.rejected_logps
        ref_logratios = reference_inputs.chosen_logps - reference_inputs.rejected_logps

        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        loss = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )

        chosen_rewards = (
            self.beta
            * (policy_inputs.chosen_logps - reference_inputs.chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta
            * (policy_inputs.rejected_logps - reference_inputs.rejected_logps).detach()
        )

        return loss, chosen_rewards, rejected_rewards

    def _bco_pair_loss(
        self,
        policy_inputs: ChosenRejectedOutputs,
        reference_inputs: ChosenRejectedOutputs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the BCO pair loss based on trl_dpo_trainer implementation."""
        # Get the log ratios for the chosen and rejected responses
        chosen_logratios = policy_inputs.chosen_logps - reference_inputs.chosen_logps
        rejected_logratios = (
            policy_inputs.rejected_logps - reference_inputs.rejected_logps
        )

        # Compute rewards
        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        # Compute rewards mean and update running moments
        rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
        self.running.update(rewards)
        delta = self.running.mean

        # Compute BCO pair loss
        loss = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
            -(self.beta * rejected_logratios - delta)
        )

        return loss, chosen_rewards, rejected_rewards

    def _sft_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute supervised fine-tuning loss using cross-entropy.

        Args:
            logits (torch.Tensor): Model logits of shape (batch_size, seq_len, vocab_size).
            labels (torch.Tensor): Target labels of shape (batch_size, seq_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - loss: The SFT loss.
                - chosen_rewards: Zero tensor (SFT doesn't have preference rewards).
                - rejected_rewards: Zero tensor (SFT doesn't have preference rewards).
        """
        # Shift logits and labels for next token prediction
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        # Flatten for cross entropy
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        # Compute cross entropy loss with ignore_index
        # CROSS_ENTROPY_IGNORE_IDX handles padding tokens automatically
        loss = F.cross_entropy(
            logits_flat.float(),
            labels_flat,
            ignore_index=CROSS_ENTROPY_IGNORE_IDX,
            reduction="sum",  # Use sum reduction, will normalize by token count later
        )

        # Count unmasked tokens (non-ignore tokens)
        unmasked_tokens = (labels_flat != CROSS_ENTROPY_IGNORE_IDX).sum().float()

        # Normalize loss by number of unmasked tokens
        if unmasked_tokens > 0:
            loss = loss / unmasked_tokens

        # For SFT, we don't have preference rewards, so use zeros
        # This maintains interface consistency with other loss types
        batch_size = logits.size(0)
        chosen_rewards = torch.zeros(batch_size, device=loss.device)
        rejected_rewards = torch.zeros(batch_size, device=loss.device)

        return loss, chosen_rewards, rejected_rewards
