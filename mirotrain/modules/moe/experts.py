# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Callable

import torch
from mirotrain.modules.moe import grouped_gemm_util as gg
from torch import nn
from torch.nn import functional as F


class GroupedGEMMExperts(nn.Module):
    """An efficient implementation of the Experts layer using GroupedGEMM.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_local_experts: int = 1,
        activation: Callable = F.silu,
    ):
        super().__init__()
        self.hidden_size = hidden_dim
        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        self.gate_proj = nn.Parameter(
            torch.empty(num_local_experts, hidden_dim, intermediate_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_local_experts, intermediate_dim, hidden_dim)
        )
        self.up_proj = nn.Parameter(
            torch.empty(num_local_experts, hidden_dim, intermediate_dim)
        )
        self.act_fn = activation
        self.reset_parameters()

        gg.assert_grouped_gemm_is_available()

    def reset_parameters(self) -> None:
        # Default initialization used by torch.nn.Linear
        nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ):
        """Forward step of the GroupedGEMMExperts."""
        tokens_per_expert = tokens_per_expert.to(device="cpu")

        if permuted_local_hidden_states.nelement() != 0:
            w1_output = gg.ops.gmm(
                permuted_local_hidden_states,
                self.gate_proj,
                tokens_per_expert,
                trans_b=False,
            )
            w3_output = gg.ops.gmm(
                permuted_local_hidden_states,
                self.up_proj,
                tokens_per_expert,
                trans_b=False,
            )
            intermediate_parallel = self.act_fn(w1_output, w3_output)
            w2_output = gg.ops.gmm(
                intermediate_parallel, self.down_proj, tokens_per_expert, trans_b=False
            )
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = self.gate_proj.view(self.hidden_size, -1)
            w3 = self.up_proj.view(self.hidden_size, -1)
            w2 = self.down_proj.view(-1, self.hidden_size)
            w1_output = torch.matmul(permuted_local_hidden_states, w1)
            w3_output = torch.matmul(permuted_local_hidden_states, w3)
            intermediate_parallel = self.act_fn(w1_output, w3_output)
            w2_output = torch.matmul(intermediate_parallel, w2)

        return w2_output
