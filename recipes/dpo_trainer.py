# SPDX-FileCopyrightText: 2025 MiromindAI
# SPDX-FileCopyrightText: Meta Platforms, Inc. and affiliates

# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

import datetime
import os
import sys
import time
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import mirotrain.monkey

import torch

mirotrain.monkey.patch_common()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mirotrain.datasets import padded_collate_dpo
from mirotrain.modules.attention import set_use_flash_attention
from mirotrain.modules.transformer import MoETransformerSelfAttentionLayer
from mirotrain.modules.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_group,
    ulysses_pad_and_slice_inputs,
)

# Import mirotrain training modules for Ulysses SP support
from mirotrain.training import ParallelDims, shard_model
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, modules, rlhf, training, utils
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets import ConcatDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.rlhf import ChosenRejectedOutputs
from torchtune.training import (
    disable_dropout,
    DummyProfiler,
    PROFILER_KEY,
    VALID_BACKENDS_FOR_MEMORY_STATS,
)
from torchtune.training.checkpointing._checkpoint_client import (
    CheckpointClient,
    TrainingProgress,
)
from torchtune.training.lr_schedulers import get_lr
from torchtune.utils import get_world_size_and_rank


class FullDPORecipeDistributed(FTRecipeInterface):
    """
    Full DPO finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe supports
    distributed training and can be run on a single node (1 to 8 GPUs).

    Features:
        - FSDP. Supported using PyTorch's FSDP APIs. CPU offload of parameters, gradients, and optimizer states
            is supported via ``fsdp_cpu_offload``. Resharding of parameters after the forward pass is
            done by default (corresponding to FULL_SHARD sharding strategy), but can be disabled by setting the config
            ``fsdp_reshard_after_forward`` to False (this corresponds to SHARD_GRAD_OP sharding strategy).
            DDP is currently not supported. Training on CPU is not supported.

        - Activation Checkpointing. This can be controlled using the ``enable_activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Activation Offloading. This can be controlled using the ``enable_activation_offloading``
            flag. Activation offloading is a technique similar to activations checkpointing that helps
            reduce the memory footprint to prevent OOMs on CUDA and enable bigger batches. Where activations
            checkpointing drops the activation in the forward to recompute it later in the backward,
            activations offloading will drop the activation in the forward to the CPU and bring it
            back during the backward pass. As always, there is a tradeoff--these savings in memory can
            come at the cost of training performance and CPU resources. To recover some runtime cost,
            we've added an option to enable offloading on a different stream to permit overlapping with
            the computation. This option is currently only available on PyTorch 2.5 or later and will
            be enabled by default if an acceptable torch version is found. Activation offloading can be
            used in conjunction with activation checkpointing.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * number of GPUs * gradient accumulation steps.

            For example: with batch_size=1, nproc_per_node=2 and gradient_accumulation_steps=32 we get a
            total batch size of 64.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Optimizer state and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training.

            Resuming training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/deep_dives/checkpointer.html).

        - Gradient Clipping. Gradient clipping is supported using the ``clip_grad_norm`` flag. By default,
            ``clip_grad_norm`` is set to ``None``. If you only want to log the grad norm, you can set
            ``clip_grad_norm='inf'``.

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

        - Memory Profiling. Memory tracing can be enabled using the ``enable_memory_trace`` flag.
            This will record CUDA memory usage during training and save it to a file for analysis.

    The following losses are supported in this recipe:
        - :class:`~torchtune.modules.rlhf.loss.DPOLoss`: Direct Preference Optimization (DPO).
        - :class:`~torchtune.rlhf.loss.RSOPLoss`: Rejection Sampling Optimization (RSO).



    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16 or if ``memory_trace_file_path`` is not provided
                   when ``enable_memory_trace`` is True.
        RuntimeError: If gradient clipping is enabled with optimizer in backward, or if gradient
                     accumulation is enabled with optimizer in backward, or if
                     ``enable_activation_offloading`` is True and device is not CUDA, or if
                     ``enable_activation_offloading`` is True and
                     ``enable_activation_checkpointing`` is False.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        self._logger = utils.get_logger(cfg.log_level)

        if (
            self._log_peak_memory_stats
            and self._device.type not in VALID_BACKENDS_FOR_MEMORY_STATS
        ):
            self._logger.info(
                f"log_peak_memory_stats was set to True; however, training device is not in {VALID_BACKENDS_FOR_MEMORY_STATS}."
                "Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Set up the backend for distributed training (NCCL, GLOO, etc.)
        self._enable_async_checkpointing = cfg.get("enable_async_checkpointing", False)
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        self.distributed_backend = training.get_distributed_backend(
            cfg.device, offload_ops_to_cpu=True
        )
        init_process_group(
            self.distributed_backend, timeout=datetime.timedelta(seconds=1800)
        )
        self._checkpoint_client = CheckpointClient(cfg)

        self.world_size, self.rank = get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0

        # Initialize distributed variables for Ulysses SP
        self.tp_degree = 1
        self.tp_plan = None
        data_shard = cfg.get("data_parallel_shard_dim", -1)  # -1 means to infer
        data_replicate = cfg.get("data_parallel_replicate_dim", 1)
        self.ulysses_sp_degree = cfg.get("ulysses_sequence_parallel_size", 1)
        self.expert_parallel_size = cfg.get("moe_expert_parallel_size", 1)
        # More robust MoE detection: check if model_type contains "moe" (case insensitive)
        model_type = cfg.checkpointer.get("model_type", "")
        self.use_moe = "moe" in model_type.lower()

        # Set up n-d device mesh
        self.parallel_dims = ParallelDims(
            dp_replicate=data_replicate,
            dp_shard=data_shard,
            tp=1,
            ulysses_sp=self.ulysses_sp_degree,
            moe_ep=self.expert_parallel_size,
            world_size=self.world_size,
        )
        self.world_mesh, self.sp_mesh, self.ep_mesh = self.parallel_dims.build_mesh(
            device_type=cfg.device
        )
        if self.parallel_dims.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            self.dp_degree, self.dp_rank = (
                dp_mesh.size(),
                dp_mesh.get_local_rank(),
            )
        else:
            self.dp_degree, self.dp_rank = 1, 0

        self.real_dp_size = self.dp_degree // self.ulysses_sp_degree
        self.real_dp_rank = self.dp_rank // self.ulysses_sp_degree

        # MoE cfg
        self._router_aux_loss_coef = cfg.model.get("router_aux_loss_coef", 0.0)
        self.use_grouped_gemm = cfg.model.get("use_grouped_gemm", True)
        self.use_fused_adamw = cfg.optimizer.get("fused", True)

        if self.use_moe and self.expert_parallel_size > 1:
            assert (
                self.use_grouped_gemm is True
            ), "Must set use_grouped_gemm to True if moe_expert_parallel_size > 1"
            if self.use_fused_adamw:
                self.use_fused_adamw = False
                cfg.optimizer["fused"] = False

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        self._clip_grad_norm = cfg.get("clip_grad_norm", None)

        self._use_flash_attention = cfg.get("use_flash_attention", True)
        set_use_flash_attention(self._use_flash_attention)

        if self.ulysses_sp_degree > 1:
            assert (
                self._use_flash_attention
            ), "ulysses sequence parallel is not supported without flash attention, please set `use_flash_attention: True`."

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )

        # Disable activation offloading for MoE models as it's not compatible
        if self.use_moe and self._enable_activation_offloading:
            utils.log_rank_zero(
                self._logger,
                "Activation offloading is not compatible with MoE models. Disabling activation offloading.",
            )
            self._enable_activation_offloading = False

        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif self._enable_activation_checkpointing:
            utils.log_rank_zero(
                self._logger,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        # These attributes constitute the recipe state and are updated by ``load_checkpoint``
        # when ``resume_from_checkpoint`` is ``True``
        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

        # Memory tracing configuration
        self._enable_memory_trace = cfg.get("enable_memory_trace", False)
        self._memory_trace_start_step = cfg.get("memory_trace_start_step", 2)
        self._memory_trace_end_step = cfg.get("memory_trace_end_step", None)
        self._memory_trace_file_path = cfg.get("memory_trace_file_path", None)

        # MPO configuration is now handled by the loss function itself

        # Validate memory tracing configuration
        if self._enable_memory_trace:
            if self._memory_trace_file_path is None:
                raise ValueError(
                    "memory_trace_file_path must be provided when enable_memory_trace is True"
                )

            # Ensure output directory exists
            os.makedirs(os.path.dirname(self._memory_trace_file_path), exist_ok=True)

            if self._memory_trace_end_step is None:
                self._memory_trace_end_step = self._memory_trace_start_step + 1

            self._logger.info(
                f"Memory tracing enabled. Trace will be saved to: {self._memory_trace_file_path}"
            )
            self._logger.info(
                f"Memory trace will start at step {self._memory_trace_start_step}"
            )
            self._logger.info(
                f"Memory trace will end at step {self._memory_trace_end_step}"
            )

    def _save_memory_trace(self) -> None:
        """
        Save the CUDA memory trace to a file using torch.cuda.memory._dump_snapshot().
        This method should be called after stopping memory recording.
        """
        if not self._enable_memory_trace:
            return

        try:
            # Use the official PyTorch method to dump snapshot directly to file
            # This is the recommended pattern: _dump_snapshot() saves a pickled snapshot
            rank_trace = self._memory_trace_file_path + f"_{self.rank}.pickle"
            torch.cuda.memory._dump_snapshot(rank_trace)
            self._logger.info(f"Memory trace saved to: {rank_trace}")

        except Exception as e:
            if self._is_rank_zero:
                self._logger.warning(f"Failed to save memory trace: {e}")

    def start_memory_trace(self) -> None:
        """
        Manually start memory tracing.

        The trace will be saved to the default path from config.
        Recommended to use .pickle extension for the file.
        """
        rank_trace = self._memory_trace_file_path + f"_{self.rank}.pickle"
        torch.cuda.memory._record_memory_history()
        self._logger.info(
            f"Started manual memory tracing. Trace will be saved to: {rank_trace}"
        )

    def _load_ref_checkpoint(self, cfg_ref_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the reference model checkpoint state from file.
        """
        _ref_checkpointer = config.instantiate(
            cfg_ref_checkpointer, should_load_recipe_state=False
        )
        checkpoint_dict = _ref_checkpointer.load_checkpoint()
        return checkpoint_dict[training.MODEL_KEY]

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)

            # log config with parameter override
            self._metric_logger.log_config(cfg)

        # Load the base model
        checkpoint_dict = self._checkpoint_client.load_base_checkpoint()
        ref_checkpoint_dict = self._load_ref_checkpoint(cfg.ref_checkpointer)

        self._compile = cfg.get("compile", False)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
        )

        # TODO (@SalmanMohammadi) investigate TP for ref model
        self._ref_model = self._setup_reference_model(
            cfg_model=cfg.model,
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            model_state_dict=ref_checkpoint_dict,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
        )

        # Enable CPU offload for chunks if model supports it
        if hasattr(self._model, "set_cpu_offload_for_chunks"):
            self._model.skip_output_layer = True
            self._ref_model.skip_output_layer = True
            self._model.set_cpu_offload_for_chunks(False)
            self._ref_model.set_cpu_offload_for_chunks(False)
            self._model.set_num_output_chunks(cfg.get("unembed_chunk_size", 4))
            self._ref_model.set_num_output_chunks(cfg.get("unembed_chunk_size", 4))

        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if training.OPT_KEY in checkpoint_dict
                else None
            ),
        )

        if self._resume_from_checkpoint:
            # If async checkpointing is enabled, intermediate checkpoints are saved asynchronously
            # using the DistributedCheckpointer.
            # Therefore the recipe needs to load the distributed checkpoint to restore the training
            # progress.
            if self._enable_async_checkpointing:
                checkpoint_dict = self._checkpoint_client.load_distributed_checkpoint(
                    self._model,
                    self._optimizer,
                )

            self._update_recipe_state(checkpoint_dict)

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)

        # Liger DPO requires hidden states, not logprobs
        # self._loss_fn.set_model_output(self._model, self._ref_model)

        if self._compile:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        if self._is_rank_zero:
            self._logger.info("Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after all of these are initialized
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.
        #
        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader, the max_steps_per_epoch param set by the user and the
        # gradient_accumulation_steps param. This value is used for logging and tracking
        # training state. The computation should happen after the dataloader has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch
        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )
        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler
        """
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        if self._is_rank_zero:
            self._logger.info(f" Profiler config after instantiation: {profiler_cfg}")

            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: Dict[str, Any],
        custom_sharded_layers: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
        """

        utils.log_rank_zero(
            self._logger,
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        # original activation checkpointing (full) - flip the condition above
        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model,
                auto_wrap_policy={
                    MoETransformerSelfAttentionLayer,
                    modules.TransformerSelfAttentionLayer,
                },
            )

        # For FSDP sharding
        fsdp_shard_conditions = [
            partial(
                training.get_shard_conditions,
                names_to_match=custom_sharded_layers,
            )
        ]

        if self.parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard")
        else:
            dp_mesh_dim_names = ("dp_shard",)

        shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=reshard_after_forward,
            dp_mesh=self.world_mesh[dp_mesh_dim_names],
            ep_mesh=self.ep_mesh,
        )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_state_dict,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        # activation offloading - for MoE models, create a dummy context manager
        if self.use_moe:
            # Create a dummy context manager for MoE models to avoid activation offloading issues
            self.activations_handling_ctx = nullcontext()
        else:
            self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
                model, enable_activation_offloading
            )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        utils.log_rank_zero(
            self._logger,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )

        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # disabling dropout if found - non-determinism leads to issues in e.g. comparing logprobs
        # between ref policy and current policy
        disable_dropout(model)

        # synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_reference_model(
        self,
        cfg_model: DictConfig,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: Dict[str, Any],
        custom_sharded_layers: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Similar to `self._setup_model`:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``

        Additionally, since the reference model is inference-only, we omit some training-specific
        optimizations.
        """

        utils.log_rank_zero(
            self._logger,
            "FSDP is enabled. Instantiating reference model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        # For FSDP sharding
        fsdp_shard_conditions = [
            partial(
                training.get_shard_conditions,
                names_to_match=custom_sharded_layers,
            )
        ]

        if self.parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard")
        else:
            dp_mesh_dim_names = ("dp_shard",)

        shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=reshard_after_forward,
            dp_mesh=self.world_mesh[dp_mesh_dim_names],
            ep_mesh=self.ep_mesh,
        )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_state_dict,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        utils.log_rank_zero(
            self._logger,
            f"Instantiating reference model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )

        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # disabling dropout if found - non-determinism leads to issues in e.g. comparing logprobs
        # between ref policy and current policy
        disable_dropout(model)

        for p in model.parameters():
            p.requires_grad = False

        model.eval()

        # synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        opt_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                self._model,
                optimizer,
                opt_state_dict,
                self._device,
            )

        utils.log_rank_zero(self._logger, "Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        if self._is_rank_zero:
            self._logger.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> StatefulDataLoader:
        """
        All data related setup happens here. This recipe currently supports only
        map-style datasets. If a state_dict is provided (meaning we are resuming a training run),
        it is loaded into the dataloader.
        """

        class LetRankZeroLoadDatasetFirst:
            def __init__(self, is_rank_zero: bool, logger):
                self.is_rank_zero = is_rank_zero
                self.logger = logger

            def __enter__(self):
                if self.is_rank_zero:
                    self.logger.info("Start to load raw dataset on rank 0")
                else:
                    torch.distributed.barrier()
                    self.logger.info("Start to load raw dataset on non-zero ranks")

            def __exit__(self, *args, **kwargs):
                if self.is_rank_zero:
                    self.logger.info("Finish loading raw dataset on rank 0")
                    torch.distributed.barrier()

        def _load_dataset(cfg: DictConfig):
            with LetRankZeroLoadDatasetFirst(self._is_rank_zero, self._logger):
                return config.instantiate(cfg, tokenizer=self._tokenizer)

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                _load_dataset(single_cfg_dataset) for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = _load_dataset(cfg_dataset)

        sampler = StatefulDistributedSampler(
            ds, num_replicas=self.real_dp_size, rank=self.real_dp_rank, shuffle=shuffle
        )

        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=partial(
                padded_collate_dpo,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
            ),
        )

        if self._is_rank_zero:
            self._logger.info("Dataset and Sampler are initialized.")

        return dataloader

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        self._checkpoint_client.save_checkpoint(
            model=self._model,
            optimizer=self._optimizer,
            training_progress=TrainingProgress(
                seed=self.seed,
                epochs_run=self.epochs_run,
                total_epochs=self.total_epochs,
                max_steps_per_epoch=self.max_steps_per_epoch,
            ),
            epoch=epoch,
        )

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        activations_handling: Optional[bool] = True,
    ) -> Tuple[ChosenRejectedOutputs, Optional[torch.Tensor]]:
        """Forward pass with CPU offload for memory efficiency."""
        concatenated_input_ids, concatenated_labels, attention_mask = batch
        concatenated_input_ids = concatenated_input_ids.to(self._device)
        concatenated_labels = concatenated_labels.to(self._device)
        attention_mask = attention_mask.to(self._device)

        # formed by concatenating an equal number of "chosen" and "rejected".
        len_chosen = concatenated_input_ids.shape[0] // 2

        # pad and slice the inputs if sp > 1
        if self.ulysses_sp_degree > 1:
            (
                concatenated_input_ids,
                input_pos,
                pad_size,
            ) = ulysses_pad_and_slice_inputs(
                concatenated_input_ids,
                sp_size=self.ulysses_sp_degree,
            )
            if self.use_moe:
                (attention_mask, _, _,) = ulysses_pad_and_slice_inputs(
                    attention_mask,
                    sp_size=self.ulysses_sp_degree,
                )

        # forward
        def forward_call():
            if self.use_moe:
                return model(concatenated_input_ids, attention_mask=attention_mask)
            else:
                return model(concatenated_input_ids)

        # For MoE models, disable activation handling as it's not compatible
        if self.use_moe:
            activations_handling = False

        # For MoE models, don't use activation offloading context as it's not compatible
        if activations_handling and not self.use_moe:
            with self.activations_handling_ctx:
                outputs = forward_call()
        else:
            outputs = forward_call()

        # Handle MoE auxiliary loss
        moe_aux_loss = None
        if isinstance(outputs, tuple):
            outputs, moe_aux_loss = outputs

        # gather output if sp > 1
        if self.ulysses_sp_degree > 1:
            outputs = gather_outpus_and_unpad(
                outputs, gather_dim=1, unpad_dim=1, padding_size=pad_size
            )

        # Use CPU offload for memory efficiency
        all_logits = model.chunked_output_with_cpu_offload(outputs)
        if isinstance(all_logits, list):
            all_logits_cpu = torch.cat([chunk.cpu() for chunk in all_logits], dim=1)
        else:
            all_logits_cpu = all_logits.cpu()
        concatenated_labels_cpu = concatenated_labels.cpu()
        del all_logits

        chosen_log_probs = rlhf.get_batch_log_probs(
            all_logits_cpu[:len_chosen],
            concatenated_labels_cpu[:len_chosen],
            return_average_logprobs=False,
        )
        rejected_log_probs = rlhf.get_batch_log_probs(
            all_logits_cpu[len_chosen:],
            concatenated_labels_cpu[len_chosen:],
            return_average_logprobs=False,
        )
        chosen_log_probs = chosen_log_probs.to(self._device)
        rejected_log_probs = rejected_log_probs.to(self._device)
        chosen_logits = all_logits_cpu[:len_chosen]
        rejected_logits = all_logits_cpu[len_chosen:]

        del concatenated_labels_cpu

        return (
            ChosenRejectedOutputs(
                chosen_log_probs, rejected_log_probs, chosen_logits, rejected_logits
            ),
            moe_aux_loss,
        )

    def _process_batch(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """Process batch with CPU offload for memory efficiency."""
        # Forward pass for policy model
        (
            policy_outputs,
            policy_moe_aux_loss,
        ) = self.concatenated_forward(self._model, batch)

        # Get logits mean for metrics (compute on CPU, then move mean to GPU for distributed communication)
        policy_chosen_logits_mean = (
            policy_outputs.chosen_logits.detach().mean().to(self._device)
        )
        policy_rejected_logits_mean = (
            policy_outputs.rejected_logits.detach().mean().to(self._device)
        )
        policy_chosen_logps = policy_outputs.chosen_logps
        policy_rejected_logps = policy_outputs.rejected_logps

        # Extract needed data and delete intermediate variables to save GPU memory
        policy_logps_for_loss = (
            policy_outputs.chosen_logps,
            policy_outputs.rejected_logps,
        )

        # Get chosen_logits before deleting policy_outputs
        chosen_logits = policy_outputs.chosen_logits
        chosen_labels = batch[1][
            : batch[0].shape[0] // 2
        ]  # Get chosen labels from batch

        # Delete policy outputs to free GPU memory early
        del policy_outputs

        # Forward pass for reference model
        with torch.no_grad():
            (reference_outputs, reference_moe_aux_loss,) = self.concatenated_forward(
                self._ref_model, batch, activations_handling=False
            )

        # Extract needed data and delete reference outputs to save GPU memory
        reference_logps_for_loss = (
            reference_outputs.chosen_logps,
            reference_outputs.rejected_logps,
        )
        del reference_outputs

        # Create minimal inputs for loss computation
        policy_inputs_minimal = ChosenRejectedOutputs(
            chosen_logps=policy_logps_for_loss[0],
            rejected_logps=policy_logps_for_loss[1],
            chosen_logits=torch.zeros(1),  # Dummy, not used in loss
            rejected_logits=torch.zeros(1),  # Dummy, not used in loss
        )

        reference_inputs_minimal = ChosenRejectedOutputs(
            chosen_logps=reference_logps_for_loss[0],
            rejected_logps=reference_logps_for_loss[1],
            chosen_logits=torch.zeros(1),  # Dummy, not used in loss
            rejected_logits=torch.zeros(1),  # Dummy, not used in loss
        )

        # Compute loss using the configured loss function
        # Call the loss function directly - DPOLoss now supports multiple loss types
        loss_result = self._loss_fn(
            policy_inputs_minimal,
            reference_inputs_minimal,
            chosen_logits,
            chosen_labels,
        )

        # Unpack loss result - DPOLoss returns (total_loss, chosen_rewards, rejected_rewards, loss_dict)
        total_loss, chosen_rewards, rejected_rewards, loss_dict = loss_result

        # Clean up temporary variables
        del policy_logps_for_loss, reference_logps_for_loss
        del policy_inputs_minimal, reference_inputs_minimal

        # Extract individual losses from loss_dict for metrics
        # loss_dict format: {"sigmoid": {"loss": tensor, "chosen_rewards": tensor, "rejected_rewards": tensor}, ...}
        flattened_loss_dict = {"total_loss": total_loss}
        for loss_type, loss_info in loss_dict.items():
            flattened_loss_dict[f"{loss_type}_loss"] = loss_info["loss"]

        return (
            chosen_rewards,
            rejected_rewards,
            policy_chosen_logits_mean,
            policy_rejected_logits_mean,
            policy_chosen_logps,
            policy_rejected_logps,
            policy_moe_aux_loss,
            flattened_loss_dict,
        )

    def _update_running_metrics(
        self,
        running_metrics: Dict[str, torch.Tensor],
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        policy_chosen_logits_mean: torch.Tensor,
        policy_rejected_logits_mean: torch.Tensor,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        moe_aux_loss: Optional[torch.Tensor],
        flattened_loss_dict: Dict[str, torch.Tensor],
        scaling_factor: float,
    ) -> None:
        """
        Update running metrics with current batch results.
        """
        running_metrics["rewards/chosen"] += (
            scaling_factor * chosen_rewards.mean()
        )
        running_metrics["rewards/rejected"] += (
            scaling_factor * rejected_rewards.mean()
        )
        running_metrics["rewards/accuracies"] += (
            scaling_factor * (chosen_rewards > rejected_rewards).float().mean()
        )
        running_metrics["log_probs/chosen"] += (
            scaling_factor * policy_chosen_logps.detach().mean()
        )
        running_metrics["log_probs/rejected"] += (
            scaling_factor * policy_rejected_logps.detach().mean()
        )
        running_metrics["logits/chosen"] += (
            scaling_factor * policy_chosen_logits_mean
        )
        running_metrics["logits/rejected"] += (
            scaling_factor * policy_rejected_logits_mean
        )

        # Track MoE auxiliary loss for logging
        if moe_aux_loss is not None:
            running_metrics["moe_aux_loss"] += (
                scaling_factor * moe_aux_loss.detach()
            )

        # Track individual losses from flattened_loss_dict dynamically
        for loss_name, loss_value in flattened_loss_dict.items():
            if loss_name != "total_loss":  # Skip total_loss as it's already tracked
                metric_key = f"loss/{loss_name}"
                if metric_key not in running_metrics:
                    running_metrics[metric_key] = torch.tensor(
                        0.0, device=self._device
                    )
                running_metrics[metric_key] += (
                    scaling_factor * loss_value.detach().mean()
                )

    def _log_per_step_metrics(
        self,
        running_loss: torch.Tensor,
        running_metrics: Dict[str, torch.Tensor],
        num_tokens: torch.Tensor,
        time_per_step: float,
    ) -> None:
        """
        Log per-step metrics to the metric logger.
        """
        if (
            self.global_step % self._log_every_n_steps == 0
            and self._is_rank_zero
        ):
            log_dict = {
                "loss": running_loss.detach().item(),
                "lr": get_lr(self._optimizer),
                # Use dp_degree instead of real_dp_size to avoid double division by ulysses_sp_degree
                # Since num_tokens already accounts for the tokens processed by this GPU in SP
                "tokens_per_second_per_gpu": num_tokens
                / (time_per_step * self.dp_degree),
                "rewards/chosen": running_metrics["rewards/chosen"].cpu(),
                "rewards/rejected": running_metrics[
                    "rewards/rejected"
                ].cpu(),
                "rewards/accuracies": running_metrics[
                    "rewards/accuracies"
                ].cpu(),
                "rewards/margins": (
                    running_metrics["rewards/chosen"]
                    - running_metrics["rewards/rejected"]
                ).cpu(),
                "log_probs/chosen": running_metrics[
                    "log_probs/chosen"
                ].cpu(),
                "log_probs/rejected": running_metrics[
                    "log_probs/rejected"
                ].cpu(),
                "logits/chosen": running_metrics["logits/chosen"].cpu(),
                "logits/rejected": running_metrics["logits/rejected"].cpu(),
            }

            # Add MoE auxiliary loss to logging if present
            if running_metrics["moe_aux_loss"].item() != 0:
                log_dict["moe_aux_loss"] = running_metrics[
                    "moe_aux_loss"
                ].cpu()

            # Add dynamic loss metrics to logging
            for key, value in running_metrics.items():
                if key.startswith("loss/"):
                    log_dict[key] = value.cpu()

            if self._log_peak_memory_stats:
                log_dict.update(
                    training.get_memory_stats(device=self._device)
                )
            self._metric_logger.log_dict(
                log_dict,
                step=self.global_step,
            )
            if self._is_rank_zero:
                timestamp = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                log_str = (
                    f"[{timestamp}] Step {self.global_step}: "
                    + ", ".join(
                        [f"{k}={v:.8f}" for k, v in log_dict.items()]
                    )
                )
                print(log_str)

    def train(self) -> None:
        """
        The core training loop. Supports training on subsets of the dataset using the
        ``max_steps_per_epoch``.
        """
        # clean up before training begins
        training.cleanup_before_training()

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()

        # Running metrics
        running_loss = 0
        running_metrics = {
            "rewards/chosen": torch.tensor(0.0, device=self._device),
            "rewards/rejected": torch.tensor(0.0, device=self._device),
            "rewards/accuracies": torch.tensor(0.0, device=self._device),
            "log_probs/chosen": torch.tensor(0.0, device=self._device),
            "log_probs/rejected": torch.tensor(0.0, device=self._device),
            "logits/chosen": torch.tensor(0.0, device=self._device),
            "logits/rejected": torch.tensor(0.0, device=self._device),
            "moe_aux_loss": torch.tensor(0.0, device=self._device),
        }

        # Dynamic loss metrics will be populated based on loss_dict content
        # No need to pre-define specific loss types
        num_tokens = torch.tensor(0.0, device=self._device)

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):

            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True

            self._dataloader.sampler.set_epoch(curr_epoch)
            for idx, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

                # Start memory tracing if enabled and we've reached the start step
                if (
                    self._enable_memory_trace
                    and self.global_step == self._memory_trace_start_step
                ):
                    self.start_memory_trace()

                # batch is input_ids, labels, attention_mask
                # In Ulysses SP, each GPU processes only a portion of the sequence
                # So we count the actual tokens processed by this GPU
                num_tokens += torch.tensor(batch[0].numel()) / self.ulysses_sp_degree

                # Process batch with CPU offload
                (
                    chosen_rewards,
                    rejected_rewards,
                    policy_chosen_logits_mean,
                    policy_rejected_logits_mean,
                    policy_chosen_logps,
                    policy_rejected_logps,
                    policy_moe_aux_loss,
                    flattened_loss_dict,
                ) = self._process_batch(batch)

                reward_accuracies = (chosen_rewards > rejected_rewards).float()

                # Extract total_loss from flattened_loss_dict
                loss = flattened_loss_dict["total_loss"]

                # Handle MoE auxiliary loss
                moe_aux_loss = None
                if policy_moe_aux_loss is not None:
                    moe_aux_loss = policy_moe_aux_loss
                    if self.ulysses_sp_degree > 1:
                        moe_aux_loss = moe_aux_loss.detach()
                        torch.distributed.all_reduce(
                            moe_aux_loss,
                            op=torch.distributed.ReduceOp.SUM,
                            group=get_ulysses_sequence_parallel_group(),
                        )

                loss = loss / self._gradient_accumulation_steps

                # Add MoE auxiliary loss if present
                if moe_aux_loss is not None:
                    moe_aux_loss = moe_aux_loss / self._gradient_accumulation_steps
                    loss = loss + moe_aux_loss * self._router_aux_loss_coef

                # Update running metrics
                running_loss += loss
                scaling_factor = (
                    1 / self._gradient_accumulation_steps
                )  # to average out between grad_acc steps
                self._update_running_metrics(
                    running_metrics,
                    chosen_rewards,
                    rejected_rewards,
                    policy_chosen_logits_mean,
                    policy_rejected_logits_mean,
                    policy_chosen_logps,
                    policy_rejected_logps,
                    moe_aux_loss,
                    flattened_loss_dict,
                    scaling_factor,
                )

                loss.backward()

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    # Accumulate running metrics across all devices
                    torch.distributed.all_reduce(
                        running_loss, op=torch.distributed.ReduceOp.AVG
                    )
                    torch.distributed.all_reduce(num_tokens)

                    for key in running_metrics:
                        torch.distributed.all_reduce(
                            running_metrics[key], op=torch.distributed.ReduceOp.AVG
                        )
                    if self._clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(),
                            max_norm=float(self._clip_grad_norm),
                        ).full_tensor()
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)

                    # Update the number of steps when the weights are updated
                    self.global_step += 1
                    # Step the learning rate scheduler
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()
                    loss_to_log = running_loss.detach().item()

                    # Log per-step metrics
                    self._log_per_step_metrics(
                        running_loss,
                        running_metrics,
                        num_tokens,
                        time.perf_counter() - t0,
                    )

                    # Reset running stats for the next step
                    running_loss = 0
                    running_metrics = {
                        key: torch.tensor(0.0, device=self._device)
                        for key in running_metrics
                    }
                    num_tokens = 0

                    t0 = time.perf_counter()

                    # Stop memory tracing if enabled and we've reached the end step
                    if (
                        self._enable_memory_trace
                        and self.global_step == self._memory_trace_end_step
                    ):
                        self._save_memory_trace()
                        if self._is_rank_zero:
                            self._logger.info(
                                f"Stopped memory tracing at step {self.global_step}"
                            )

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

        self._profiler.stop()

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    if not training.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    if cfg.get("fsdp_cpu_offload", False):
        # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
        # speed up when benchmarking fused AdamW on CPU
        training.set_torch_num_threads()

    config.log_config(recipe_name="FullDPORecipeDistributed", cfg=cfg)

    recipe = FullDPORecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
