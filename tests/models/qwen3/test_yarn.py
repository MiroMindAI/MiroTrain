# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
import torch

from mirotrain.models.qwen3._positional_embeddings import (
    Qwen3YarnRotaryPositionalEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    Qwen3Config,
    Qwen3RotaryEmbedding,
)


class TestQwen3YaRN:
    @pytest.fixture
    def qwen3_config(self):
        return Qwen3Config(
            num_attention_heads=8,
            head_dim=128,
            max_position_embeddings=131_072,
            rope_theta=1_000_000,
            rope_scaling={
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32_768,
            },
        )

    def init_yarn(self, qwen3_config: Qwen3Config):
        yarn_hf = Qwen3RotaryEmbedding(config=qwen3_config)
        yarn_tune = Qwen3YarnRotaryPositionalEmbedding(
            dim=qwen3_config.head_dim,
            max_position_embeddings=qwen3_config.max_position_embeddings,
            base=qwen3_config.rope_theta,
            rope_scaling=qwen3_config.rope_scaling,
        )
        return yarn_hf, yarn_tune

    def init_packed_inputs(
        self,
        qwen3_config: Qwen3Config,
        batch_size: int,
        num_packed_samples: int,
        dtype: torch.dtype,
    ):
        q_packed = torch.randn(
            batch_size,
            qwen3_config.max_position_embeddings,
            qwen3_config.num_attention_heads,
            qwen3_config.head_dim,
            dtype=dtype,
            requires_grad=True,
        )
        k_packed = torch.randn(
            batch_size,
            qwen3_config.max_position_embeddings,
            qwen3_config.num_attention_heads,
            qwen3_config.head_dim,
            dtype=dtype,
            requires_grad=True,
        )
        seq_len = qwen3_config.max_position_embeddings // num_packed_samples
        position_ids = torch.arange(seq_len).repeat(batch_size, num_packed_samples)
        return q_packed, k_packed, position_ids

    # Inputs configs
    batch_size_list = [1, 3]
    num_packed_samples_list = [1, 2]
    dtypes_list = [torch.bfloat16, torch.float32]
    inputs_configs = list(
        itertools.product(batch_size_list, num_packed_samples_list, dtypes_list)
    )
    inputs_config_ids = [
        f"batch_size_{batch_size}-num_packed_samples_{num_packed_samples}-dtype_{dtype}"
        for batch_size, num_packed_samples, dtype in inputs_configs
    ]

    @pytest.mark.parametrize(
        "batch_size, num_packed_samples, dtype",
        inputs_configs,
        ids=inputs_config_ids,
    )
    def test_yarn_init_fwd_bwd(
        self,
        qwen3_config: Qwen3Config,
        batch_size: int,
        num_packed_samples: int,
        dtype: torch.dtype,
    ):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            # NOTE REGARDING ASSERTIONS:
            # Since we migrated HuggingFace's YaRN implementation to torchtune,
            # both implementations are expected to produce *identical* outputs.
            # Therefore, all subsequent tensor comparisons in this test method
            # use exact equality (==) instead of torch.allclose().

            # Test YaRN initialization
            yarn_hf, yarn_tune = self.init_yarn(qwen3_config)
            assert (yarn_hf.inv_freq == yarn_tune.theta).all()

            # Test YaRN forward

            # Initialize packed inputs
            q, k, position_ids = self.init_packed_inputs(
                qwen3_config, batch_size, num_packed_samples, dtype
            )

            # hf fwd
            q_hf = q.transpose(1, 2)
            k_hf = k.transpose(1, 2)
            cos_hf, sin_hf = yarn_hf(q, position_ids=position_ids)
            q_out_hf, k_out_hf = apply_rotary_pos_emb(q_hf, k_hf, cos_hf, sin_hf)
            q_out_hf = q_out_hf.transpose(1, 2)
            k_out_hf = k_out_hf.transpose(1, 2)

            # tune fwd
            q_out_tune = yarn_tune(q, input_pos=position_ids)
            k_out_tune = yarn_tune(k, input_pos=position_ids)

            assert (q_out_hf == q_out_tune).all()
            assert (k_out_hf == k_out_tune).all()

            # Test YaRN backward

            # hf bwd
            loss_hf = q_out_hf.sum() + k_out_hf.sum()
            loss_hf.backward()
            q_grad_hf = q.grad.clone()
            k_grad_hf = k.grad.clone()

            q.grad.zero_()
            k.grad.zero_()

            # tune bwd
            loss_tune = q_out_tune.sum() + k_out_tune.sum()
            loss_tune.backward()
            q_grad_tune = q.grad
            k_grad_tune = k.grad

            assert (q_grad_hf == q_grad_tune).all()
            assert (k_grad_hf == k_grad_tune).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
