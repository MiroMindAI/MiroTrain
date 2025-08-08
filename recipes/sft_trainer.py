# SPDX-FileCopyrightText: 2025 MiromindAI
# SPDX-FileCopyrightText: Meta Platforms, Inc. and affiliates

# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

import datetime
import logging
import os
import sys
import time

import mirotrain.monkey

mirotrain.monkey.patch_common()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, List, Optional, Union
from warnings import warn

import torch
from mirotrain.datasets import (
    padded_collate_packed,
    StatefulDistributedStreamingPackedDataset,
)
from mirotrain.modules.attention import set_use_flash_attention
from mirotrain.modules.transformer import MoETransformerSelfAttentionLayer
from mirotrain.modules.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_group,
    ulysses_pad_and_slice_inputs,
)
from mirotrain.training import (
    clip_grad_norm_,
    ParallelDims,
    scale_grads_,
    SDCheckpointClient,
    shard_model,
    STEP_KEY,
    StepCheckpointClient,
    StepTrainingProgress,
)
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.optim import Optimizer
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.datasets import ConcatDataset, PackedDataset
from torchtune.modules.embedding_utils import resize_token_embeddings
from torchtune.modules.loss import SFTLoss
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import (
    DummyProfiler,
    PROFILER_KEY,
    VALID_BACKENDS_FOR_MEMORY_STATS,
)
from torchtune.training.activations import apply_selective_activation_checkpointing
from torchtune.training.checkpointing._checkpoint_client import TrainingProgress
from torchtune.training.lr_schedulers import get_lr
from torchtune.training.quantization import (
    convert_to_float8_training,
    is_fp8_tensorwise_scaling,
)


class FullFinetuneRecipeDistributed(FTRecipeInterface):
    """
    Full finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe supports
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

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

        - Gradient Clipping. Gradient clipping is supported using the ``clip_grad_norm`` flag. By default,
            ``clip_grad_norm`` is set to ``None``. If you only want to log the grad norm, you can set
            ``clip_grad_norm='inf'``.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is bf16 but not supported by hardware;
                  if ``left_pad_sequence`` is used as data collator;
                  if ``enable_activation_offloading`` is True but device is not CUDA;
                  or if ``enable_activation_offloading`` is True while ``enable_activation_checkpointing`` is False.

    """

    def __init__(self, cfg: DictConfig) -> None:
        device_type = cfg.device
        self._device = utils.get_device(device=device_type)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # Set up the backend for distributed training (NCCL, GLOO, etc.)
        self._enable_async_checkpointing = cfg.get("enable_async_checkpointing", False)
        self._auto_resume = cfg.get("auto_resume", False)
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        self.distributed_backend = training.get_distributed_backend(
            device_type,
            offload_ops_to_cpu=self.fsdp_cpu_offload
            or self._enable_async_checkpointing
            or self._auto_resume,
        )
        init_process_group(
            self.distributed_backend, timeout=datetime.timedelta(seconds=1800)
        )

        # Initialize distributed variables
        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0
        self.tp_plan = cfg.get("tensor_parallel_plan", None)
        self.tp_degree = cfg.get("tensor_parallel_dim", 1)
        if self.tp_degree > 1 and self.tp_plan is None:
            raise ValueError(
                "Tensor Parallel plan needs to be provided when tensor parallel is enabled."
            )
        data_shard = cfg.get("data_parallel_shard_dim", -1)  # -1 means to infer
        data_replicate = cfg.get("data_parallel_replicate_dim", 1)
        self.ulysses_sp_degree = cfg.get("ulysses_sequence_parallel_size", 1)
        self.expert_parallel_size = cfg.get("moe_expert_parallel_size", 1)
        # More robust MoE detection: check if model_type contains "moe" (case insensitive)
        model_type = cfg.checkpointer.get("model_type", "")
        self.use_moe = "moe" in model_type.lower()
        self.use_grouped_gemm = cfg.model.get("use_grouped_gemm", True)
        self.use_fused_adamw = cfg.optimizer.get("fused", True)

        # Ulysses SP is not compatible with TP now
        if self.tp_degree > 1:
            assert self.ulysses_sp_degree == 1, "Ulysses SP is not compatible with TP"
            assert self.expert_parallel_size == 1, "MoE EP is not compatible with TP"

        if self.use_moe and self.expert_parallel_size > 1:
            assert (
                self.use_grouped_gemm is True
            ), "Must set use_grouped_gemm to True if moe_expert_parallel_size > 1"
            # assert self.use_fused_adamw is False, "Must use unfused adamw if expert_parallel_size > 1"
            if self.use_fused_adamw:
                self.use_fused_adamw = False
                cfg.optimizer["fused"] = False

        # Set up n-d device mesh
        self.parallel_dims = ParallelDims(
            dp_replicate=data_replicate,
            dp_shard=data_shard,
            tp=self.tp_degree,
            ulysses_sp=self.ulysses_sp_degree,
            moe_ep=self.expert_parallel_size,
            world_size=self.world_size,
        )
        self.world_mesh, self.sp_mesh, self.ep_mesh = self.parallel_dims.build_mesh(
            device_type=device_type
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

        self._use_flash_attention = cfg.get("use_flash_attention", True)
        set_use_flash_attention(self._use_flash_attention)

        if self.ulysses_sp_degree > 1:
            assert (
                self._use_flash_attention
            ), "ulysses sequence parallel is not supported without flash attention, please set `use_flash_attention: True`."

        # Logging attributes
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

        # MoE cfg
        self._router_aux_loss_coef = cfg.model.get("router_aux_loss_coef", 0.0)

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._checkpoint_steps_for_auto_resume = cfg.get(
            "checkpoint_steps_for_auto_resume"
        )
        if self._checkpoint_steps_for_auto_resume is None and self._auto_resume:
            self._checkpoint_steps_for_auto_resume = 50
            utils.log_rank_zero(
                self._logger,
                f"checkpoint_steps_for_auto_resume is not set, using default value {self._checkpoint_steps_for_auto_resume}",
            )
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._checkpoint_client = SDCheckpointClient(cfg)
        self._step_checkpoint_client = StepCheckpointClient(cfg.checkpointer.output_dir)
        self._enable_fp8_training = cfg.get("enable_fp8_training", False)
        self._fp8_recipe_name = cfg.get("fp8_recipe_name", None)

        self._run_val_every_n_steps = cfg.get("run_val_every_n_steps", None)
        if self._run_val_every_n_steps is not None:
            assert (
                cfg.get("dataset_val") is not None
            ), "run_val_every_n_steps is set but dataset_val is not configured"

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        self._activation_offloading_use_streams = cfg.get(
            "activation_offloading_use_streams", True
        )
        if self._activation_offloading_use_streams and self.parallel_dims.tp_enabled:
            warn(
                message=(
                    "Using activation offloading with streams is not advised in tensor parallel, and may "
                    "cause unstable training. It is advised to set activation_offloading_use_streams: False"
                )
            )
        
        # Disable activation offloading for MoE models as it's not compatible
        if self.use_moe and self._enable_activation_offloading:
            utils.log_rank_zero(
                self._logger,
                "Activation offloading is not compatible with MoE models. Disabling activation offloading.",
            )
            self._enable_activation_offloading = False
            
        if self._enable_activation_offloading:
            if device_type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(
                self._logger,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

        # Memory tracing configuration
        self._enable_memory_trace = cfg.get("enable_memory_trace", False)
        self._memory_trace_start_step = cfg.get("memory_trace_start_step", 5)
        self._memory_trace_end_step = cfg.get("memory_trace_end_step", None)
        self._memory_trace_file_path = cfg.get("memory_trace_file_path", None)

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

            if self._is_rank_zero:
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
            if self._is_rank_zero:
                torch.cuda.memory._dump_snapshot(self._memory_trace_file_path)
                self._logger.info(
                    f"Memory trace saved to: {self._memory_trace_file_path}"
                )

        except Exception as e:
            if self._is_rank_zero:
                self._logger.warning(f"Failed to save memory trace: {e}")

    def start_memory_trace(self) -> None:
        """
        Manually start memory tracing.

        The trace will be saved to the default path from config.
        Recommended to use .pickle extension for the file.
        """
        if self._is_rank_zero:
            torch.cuda.memory._record_memory_history()
            self._logger.info(
                f"Started manual memory tracing. Trace will be saved to: {self._memory_trace_file_path}"
            )

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
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, lr scheduler, sampler, and dataloader.
        """
        if self.fsdp_cpu_offload:
            # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
            # speed up when benchmarking fused AdamW on CPU
            training.set_torch_num_threads()

        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            # log config with parameter override
            self._metric_logger.log_config(cfg)

        # Load the base model
        if self._is_rank_zero:
            self._logger.info(
                "Start to Load Checkpoint, it may take some time for large models"
            )
        checkpoint_dict = self._checkpoint_client.load_base_checkpoint()

        compile = cfg.get("compile")
        compile_bool = bool(compile)
        self._compile_backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")

        self._compile_model = compile_bool
        self._compile_loss = compile_bool
        self._compile_optimizer_step = compile_bool
        self._compile_scale_grads = compile_bool
        if isinstance(compile, DictConfig):
            self._compile_model = compile.get("model", True)
            self._compile_loss = compile.get("loss", True)
            self._compile_optimizer_step = compile.get("optimizer_step", False)
            self._compile_scale_grads = compile.get("scale_grads", True)

        # This indirection is needed to apply torch.compile to scale_grads step.
        self._grad_scaler = scale_grads_
        if self._compile_scale_grads:
            self._grad_scaler = torch.compile(
                self._grad_scaler, backend=self._compile_backend
            )

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            activation_offloading_use_streams=self._activation_offloading_use_streams,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
            ac_mode=cfg.get("ac_mode", None),
            ac_option=cfg.get("ac_option", None),
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        if cfg.get("resize_token_embeddings", False):
            resize_token_embeddings(self._model, self._tokenizer.vocab_size)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if training.OPT_KEY in checkpoint_dict
                else None
            ),
        )
        if self._compile_optimizer_step:
            self._optimizer.step = torch.compile(
                self._optimizer.step,
                backend=self._compile_backend,
            )

        if self._resume_from_checkpoint:
            # If async checkpointing is enabled, intermediate checkpoints are saved asynchronously
            # using the DistributedCheckpointer.
            # Therefore the recipe needs to load the distributed checkpoint to restore the training
            # progress.
            if self._enable_async_checkpointing:
                try:
                    checkpoint_dict = (
                        self._checkpoint_client.load_distributed_checkpoint(
                            self._model,
                            self._optimizer,
                        )
                    )
                except Exception as e:
                    self._logger.warning(
                        f"Failed to load distributed checkpoint: {e}. Training will start from the base checkpoint."
                    )

            # Update the recipe state from the checkpoint state dict.
            self._update_recipe_state(checkpoint_dict)

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        if isinstance(self._loss_fn, SFTLoss):
            self._loss_fn.set_model_output(self._model)

        if self._compile_loss:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        utils.log_rank_zero(self._logger, "Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_sft")
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
            seed=self.seed,
        )

        # Setup validation dataloader if validation dataset is provided
        self._val_dataloader = None
        if cfg.get("dataset_val") is not None:
            batch_size_val = cfg.get("batch_size_val", cfg.batch_size)
            self._val_dataloader = self._setup_data(
                cfg_dataset=cfg.dataset_val,
                batch_size=batch_size_val,
                collate_fn=collate_name,
                shuffle=False,
                seed=self.seed,
            )

        if self._auto_resume:
            checkpoint_dict = self._step_checkpoint_client.load_latest_checkpoint(
                self._model, self._optimizer, self._dataloader
            )

            if checkpoint_dict is None:
                utils.log_rank_zero(
                    self._logger,
                    "No step-based checkpoint was found in the output directory. Training will start from the base checkpoint.",
                    level=logging.WARNING,
                )
            else:
                # Update the recipe state from the checkpoint state dict.
                self._update_recipe_state(checkpoint_dict)

                self.global_step = checkpoint_dict[STEP_KEY]

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
        if not self._auto_resume:
            self.global_step = self.epochs_run * self._steps_per_epoch

        # Setup lr scheduler
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # Used to ignore labels for loss computation
        bsz_cache = (
            cfg.batch_size
            if self._val_dataloader is None
            else max(cfg.batch_size, self._val_dataloader.batch_size)
        )

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: Optional[DictConfig],
        num_training_steps: int,
        last_epoch: int,
    ) -> Optional[Optimizer]:
        """
        Set up the learning rate scheduler based on the provided configuration.

        Args:
            cfg_lr_scheduler (Optional[DictConfig]): The learning rate scheduler configuration.
            num_training_steps (int): The total number of training steps.
            last_epoch (int): The index of the last epoch.

        Returns:
            lr_scheduler (Optional[Optimizer]): The learning rate scheduler.
        """
        if cfg_lr_scheduler is None:
            if self._is_rank_zero:
                self._logger.info(
                    "No learning rate scheduler configured. Using constant learning rate."
                )
            return None

        # Instantiate the learning rate scheduler
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        if self._is_rank_zero:
            self._logger.info("Learning rate scheduler is initialized.")

        return lr_scheduler

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

        utils.log_rank_zero(
            self._logger, f" Profiler config after instantiation: {profiler_cfg}"
        )
        if self._is_rank_zero:
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
        activation_offloading_use_streams: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: Dict[str, Any],
        custom_sharded_layers: Optional[List[str]] = None,
        ac_mode: Optional[str] = None,
        ac_option: Optional[int] = None,
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
            "Distributed training is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        if self._compile_model:
            training.compile_model(model, verbose=self._is_rank_zero)

        if self._enable_fp8_training:
            # Requires https://github.com/pytorch/pytorch/pull/148922
            if torch.__version__ < "2.8.0.dev20250318":
                raise RuntimeError(
                    "Float8 fine-tuning requires PyTorch 2.8.0.dev20250318 or later."
                )
            if self.tp_plan is not None:
                raise ValueError(
                    "FP8 training does not support tensor parallelism yet. "
                    "This will be enabled in the near future."
                )
            model = convert_to_float8_training(model, self._fp8_recipe_name)

        # Apply tensor parallelism to the model
        if self.parallel_dims.tp_enabled:
            if not self.parallel_dims.dp_enabled and self.fsdp_cpu_offload:
                raise ValueError(
                    "Tensor parallelism is not supported with FSDP CPU offloading when data parallelism is disabled."
                )
            # Use the local number (num_heads, num_kv_heads, embed_dim) to account for tensor parallel
            model = training.prepare_mha_for_tp(model, self.world_mesh["tp"])
            if self.tp_plan is not None:
                self.tp_plan = config.instantiate(
                    self.tp_plan,
                    model=model,
                )
            parallelize_module(
                model,
                self.world_mesh["tp"],
                parallelize_plan=self.tp_plan,
            )

        # We currently have two versions of activation checkpointing in this recipe
        # for testing and BC purposes. ``enable_activation_checkpointing`` controls
        # the older version of AC and this behavior is unchanged
        # ac_mode and ac_option together control selective AC. This is only enabled
        # when these are set AND ``enable_activation_checkpointing`` is set to False
        # We'll clean this up as soon as testing of AC is complete
        if (not enable_activation_checkpointing) and (ac_mode is not None):
            apply_selective_activation_checkpointing(
                model,
                ac_mode,
                ac_option,
            )

        # original activation checkpointing (full) - flip the condition above
        if enable_activation_checkpointing and ac_mode is None:
            training.set_activation_checkpointing(
                model,
                auto_wrap_policy={
                    MoETransformerSelfAttentionLayer,
                    modules.TransformerSelfAttentionLayer,
                },
            )

        # Apply Fully Sharded Data Parallelism to the model
        if self.parallel_dims.dp_shard_enabled:
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
                model, enable_activation_offloading, activation_offloading_use_streams
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

        # synchronize before training begins
        torch.distributed.barrier(device_ids=[self._device.index])

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

        utils.log_rank_zero(self._logger, "Optimizer is initialized.")
        return optimizer

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
        dataloader_state_dict: Optional[Dict[str, Any]] = None,
        seed=0,
    ) -> StatefulDataLoader:
        """
        All data related setup happens here. This recipe currently supports only
        map-style datasets. If a state_dict is provided (meaning we are resuming a training run),
        it is loaded into the dataloader.
        """
        max_seq_len = self._tokenizer.max_seq_len

        class LetRankZeroLoadDatasetFirst:
            def __init__(self, is_rank_zero: bool, logger: logging.Logger):
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
                return config.instantiate(cfg, self._tokenizer)

        def _override_packed_seq_len(
            ds: ConcatDataset | PackedDataset, packed_seq_len: int
        ):
            self._logger.warning(
                f"Overriding PackedDataset.max_seq_len with packed_seq_len={packed_seq_len}"
            )
            if isinstance(ds, ConcatDataset):
                for sub_ds in ds.datasets:
                    if isinstance(sub_ds, PackedDataset):
                        sub_ds.max_seq_len = packed_seq_len
            elif isinstance(ds, PackedDataset):
                ds.max_seq_len = packed_seq_len
            else:
                raise ValueError(
                    "Packed dataset expected, but got unknown dataset type."
                )

        # ==== Load Dataset ====
        if isinstance(cfg_dataset, ListConfig):
            packed_seq_lens = {cfg.pop("packed_seq_len", 0) for cfg in cfg_dataset}
            streaming_flags = {cfg.pop("streaming_pack", False) for cfg in cfg_dataset}

            if len(packed_seq_lens) > 1:
                raise ValueError(
                    "All 'packed_seq_len' values must be the same in ConcatDataset."
                )
            if len(streaming_flags) > 1:
                raise ValueError(
                    "All 'streaming_pack' values must be the same in ConcatDataset."
                )

            packed_seq_len = packed_seq_lens.pop()
            streaming_pack = streaming_flags.pop()

            datasets = [_load_dataset(cfg) for cfg in cfg_dataset]
            ds = ConcatDataset(datasets)
            packed = getattr(ds, "packed", False)

        else:
            packed_seq_len = cfg_dataset.pop("packed_seq_len", 0)
            streaming_pack = cfg_dataset.pop("streaming_pack", False)
            ds = _load_dataset(cfg_dataset)
            packed = cfg_dataset.get("packed", False)

        # ==== Validation ====
        if packed and streaming_pack:
            raise ValueError("'packed' and 'streaming_pack' cannot both be True.")

        if packed or streaming_pack:
            if batch_size != 1:
                raise ValueError("Packed/streaming datasets require batch_size = 1.")
            if max_seq_len is None:
                raise ValueError("max_seq_len must be set when using packed dataset.")
            if packed_seq_len < max_seq_len:
                raise ValueError(
                    f"packed_seq_len ({packed_seq_len}) < max_seq_len ({max_seq_len})"
                )

        # ==== Post Processing Dataset ====
        if packed:
            _override_packed_seq_len(ds, packed_seq_len)

        if streaming_pack:
            ds = StatefulDistributedStreamingPackedDataset(
                ds,
                max_seq_len=packed_seq_len,
                num_replicas=self.real_dp_size,
                rank=self.real_dp_rank,
                seed=seed,
            )

        # ==== collate_fn ====
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")

        collate_fn_obj = _get_component_from_path(collate_fn)
        if packed or streaming_pack:
            collate = partial(
                padded_collate_packed,
                use_flash_attention=self._use_flash_attention,
            )
        else:
            collate = partial(
                collate_fn_obj,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,
                pad_to_multiple_of=self.tp_degree,
            )

        # ==== Sampler ====
        sampler = None
        if not streaming_pack:
            sampler = StatefulDistributedSampler(
                ds,
                num_replicas=self.real_dp_size,
                rank=self.real_dp_rank,
                shuffle=shuffle,
                seed=seed,
            )

        # ==== Dataloader ====
        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=False if self._use_flash_attention else True,
            # Enable prefetching when streaming_pack is enabled (prefetch_factor is 2 * num_workers by default)
            # DON'T CHANGE num_workers BEFORE CONSULTING WITH THE INFRASTRUCTURE TEAM!
            num_workers=1 if streaming_pack else 0,
        )

        return dataloader

    def _loss_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Shape [b, s], needed for the loss not the model
        labels = batch.pop("labels")

        if not self.use_moe:
            batch.pop("attention_mask", None)

        # pad and slice the inputs if sp > 1
        if self.ulysses_sp_degree > 1:
            (
                batch["tokens"],
                batch["input_pos"],
                pad_size,
            ) = ulysses_pad_and_slice_inputs(
                batch["tokens"],
                batch.pop("input_pos", None),
                sp_size=self.ulysses_sp_degree,
            )

            if self.use_moe:
                batch["attention_mask"], _, pad_size = ulysses_pad_and_slice_inputs(
                    batch["attention_mask"],
                    sp_size=self.ulysses_sp_degree,
                )

        with self.activations_handling_ctx:
            outputs = self._model(**batch)
            moe_aux_loss = None
            if isinstance(outputs, tuple):
                outputs, moe_aux_loss = outputs

        # gather output if sp > 1
        if self.ulysses_sp_degree > 1:
            outputs = gather_outpus_and_unpad(
                outputs, gather_dim=1, unpad_dim=1, padding_size=pad_size
            )

        # post process for third party loss functions
        if not isinstance(self._loss_fn, SFTLoss):
            labels = labels.reshape(-1)
            outputs = outputs.reshape(-1, outputs.size(-1))
            if isinstance(outputs, DTensor):
                outputs = outputs.full_tensor()

        # Compute loss
        loss = self._loss_fn(outputs, labels)
        # process moe aux loss
        if moe_aux_loss is not None:
            if self.ulysses_sp_degree > 1:
                moe_aux_loss = moe_aux_loss.detach()
                torch.distributed.all_reduce(
                    moe_aux_loss,
                    op=torch.distributed.ReduceOp.SUM,
                    group=get_ulysses_sequence_parallel_group(),
                )

        # free logits otherwise it peaks backward memory
        del outputs

        return loss, moe_aux_loss

    def validate(self) -> Dict[str, float]:
        """
        Run validation loop and return average validation loss.
        """
        self._model.eval()
        total_val_loss = torch.tensor(0.0, device=self._device)
        total_val_tokens = torch.tensor(0.0, device=self._device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self._val_dataloader):
                utils.batch_to_device(batch, self._device)

                # Count tokens excluding padding
                current_num_tokens = (
                    batch["labels"] != self._loss_fn.ignore_index
                ).sum()

                # Compute loss
                val_loss, _ = self._loss_step(batch) * current_num_tokens

                total_val_loss += val_loss
                total_val_tokens += current_num_tokens

        # Aggregate validation metrics across all ranks
        torch.distributed.all_reduce(total_val_loss)
        torch.distributed.all_reduce(total_val_tokens)

        avg_val_loss = (
            (total_val_loss / total_val_tokens).item()
            if total_val_tokens > 0
            else float("inf")
        )
        log_dict = {"val_loss": avg_val_loss}

        if self._is_rank_zero:
            self._logger.info(f"Validation loss: {avg_val_loss:.4f}")
            self._metric_logger.log_dict(
                log_dict,
                step=self.global_step,
            )

        self._model.train()
        return log_dict

    def train(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_unmasked_tokens = 0
        num_all_tokens = 0

        self._profiler.start()
        is_first_resuming_step = self.global_step > 0
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            if isinstance(
                self._dataloader.dataset, StatefulDistributedStreamingPackedDataset
            ):
                self._dataloader.dataset.set_epoch(curr_epoch)
                # Suppress warnings caused by the actual number of fetched packs
                # exceeding the estimated total number of packs. Refer to
                # <https://github.com/pytorch/pytorch/blob/414ad470450c654d97e73bef704a7b596b5b4cbc/torch/utils/data/dataloader.py#L738>
                self._dataloader._IterableDataset_len_called = None
            else:
                self._dataloader.sampler.set_epoch(curr_epoch)
            for idx, batch in enumerate(self._dataloader):
                if self._auto_resume:
                    # More general computation of batch index that supports both normal training
                    # and resuming from step-based checkpoints.
                    # While this statement could theoretically be moved outside the if scope,
                    # we keep it here to ensure existing torchtune workflow remains unaffected.
                    idx = self._dataloader._iterator._num_yielded - 1

                    if is_first_resuming_step:
                        is_first_resuming_step = False

                        # IMPORTANT: Any future logic added to the for loop after the step-based checkpoint save operation
                        # must be copied here to ensure perfect resuming behavior.

                        # max_steps_per_epoch limit may already be reached when resuming from a step-based checkpoint
                        if (
                            (idx + 1) // self._gradient_accumulation_steps
                            == self.max_steps_per_epoch
                        ):
                            break

                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history()

                # Start memory tracing if enabled and we've reached the start step
                if (
                    self._enable_memory_trace
                    and self.global_step == self._memory_trace_start_step
                ):
                    self.start_memory_trace()

                utils.batch_to_device(batch, self._device)

                # Calculate the number of unmasked tokens in the current batch
                # and increment the total number of tokens seen in the step
                current_num_unmasked_tokens = (
                    batch["labels"] != self._loss_fn.ignore_index
                ).sum()

                num_unmasked_tokens += current_num_unmasked_tokens

                current_num_all_tokens = torch.tensor(
                    batch["labels"].numel(), device=self._device
                )
                num_all_tokens += current_num_all_tokens

                # Loss is normalized by default so we multiply by the number of tokens
                # This way we can normalize by the total number of tokens if we're accumulating gradients
                current_loss, moe_aux_loss = self._loss_step(batch)
                current_loss *= current_num_unmasked_tokens
                if moe_aux_loss is not None:
                    current_loss = (
                        current_loss + moe_aux_loss * self._router_aux_loss_coef
                    )
                running_loss += current_loss

                current_loss.backward()
                # Optimizer step
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    # Get total number of tokens across all ranks to normalize gradients
                    torch.distributed.all_reduce(num_unmasked_tokens)
                    num_unmasked_tokens = num_unmasked_tokens / self.ulysses_sp_degree
                    torch.distributed.all_reduce(num_all_tokens)
                    num_all_tokens = num_all_tokens / self.ulysses_sp_degree
                    # This will ensure that the logged loss matches what we're optimizing
                    torch.distributed.all_reduce(running_loss)
                    running_loss = running_loss / self.ulysses_sp_degree

                    # Manually scale the gradients from unnormalized loss by total # of tokens
                    self._grad_scaler(
                        self._model.parameters(),
                        self.world_size
                        / (num_unmasked_tokens * self.ulysses_sp_degree),
                        False if self.parallel_dims.tp_enabled else None,
                    )

                    if self._clip_grad_norm is not None:
                        grad_norm = clip_grad_norm_(
                            self._model.parameters(),
                            max_norm=float(self._clip_grad_norm),
                        )
                        # If sharded, collect the DTensor here
                        if isinstance(grad_norm, DTensor):
                            grad_norm = grad_norm.full_tensor()
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    # Step the learning rate scheduler
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    # If float8 training is enabled, perform a single all-reduce to compute the
                    # scale for all float8 parameters efficiently instead of doing many small
                    # all-reduces for each parameter
                    if (
                        self._enable_fp8_training
                        and is_fp8_tensorwise_scaling(self._fp8_recipe_name)
                        and self.dp_degree > 1
                    ):
                        precompute_float8_dynamic_scale_for_fsdp(self._model)

                    loss_to_log = running_loss.detach().item() / num_unmasked_tokens

                    # Log per-step metrics
                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "lr": get_lr(self._optimizer),
                            "unmasked_tgs": (
                                num_unmasked_tokens
                                / self.parallel_dims.non_data_parallel_size
                            )
                            / (time_per_step * self.world_size),
                            "all_tgs": (
                                num_all_tokens
                                / self.parallel_dims.non_data_parallel_size
                            )
                            / (time_per_step * self.world_size),
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        if self._clip_grad_norm is not None:
                            log_dict.update({"grad_norm": grad_norm})
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

                    # Reset running stats for the next step
                    running_loss = 0
                    num_unmasked_tokens = 0
                    num_all_tokens = 0
                    t0 = time.perf_counter()

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                        and self._device.type == "cuda"
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

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

                    # Run validation after gradient update
                    if (
                        self._run_val_every_n_steps is not None
                        and self.global_step % self._run_val_every_n_steps == 0
                    ):
                        self.validate()

                    if (
                        self._auto_resume
                        and self.global_step % self._checkpoint_steps_for_auto_resume
                        == 0
                    ):
                        self._step_checkpoint_client.save_checkpoint_async(
                            model=self._model,
                            optimizer=self._optimizer,
                            training_progress=StepTrainingProgress(
                                seed=self.seed,
                                epochs_run=self.epochs_run,
                                total_epochs=self.total_epochs,
                                max_steps_per_epoch=self.max_steps_per_epoch,
                                dataloader_state_dict=self._dataloader.state_dict(),
                                global_step=self.global_step,
                            ),
                        )
                if (
                    (idx + 1) // self._gradient_accumulation_steps
                ) == self.max_steps_per_epoch:
                    break

            self.epochs_run += 1

            # Wait for the step-based checkpoint to complete its asynchronous save operation
            # before saving the epoch-based checkpoint to avoid race conditions between collective calls,
            # which could result in a collective hang.
            # Ref: <https://pytorch.org/blog/reducing-checkpointing-times/>
            if self._step_checkpoint_client._get_dcp_checkpointer().checkpoint_future:
                self._step_checkpoint_client._get_dcp_checkpointer().checkpoint_future.result()

            self._checkpoint_client.save_checkpoint(
                model=self._model,
                optimizer=self._optimizer,
                training_progress=TrainingProgress(
                    seed=self.seed,
                    epochs_run=self.epochs_run,
                    total_epochs=self.total_epochs,
                    max_steps_per_epoch=self.max_steps_per_epoch,
                    dataloader_state_dict=self._dataloader.state_dict(),
                ),
                epoch=curr_epoch,
            )

            # This code block clears the state of the dataset in the main process.
            # Currently, this operation has no effect because:
            # - With num_workers=1 (for prefetching): the dataset in the main process holds no state
            # - With persistent_workers=False: worker processes that hold state are recreated each epoch
            # However, if num_workers=0 is set (e.g. for debugging), the dataset in the main process would hold state,
            # making this code block necessary for proper cleanup.
            if isinstance(
                self._dataloader.dataset, StatefulDistributedStreamingPackedDataset
            ):
                self._dataloader.dataset.clear_all_states()

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
    config.log_config(recipe_name="FullFinetuneRecipeDistributed", cfg=cfg)
    recipe = FullFinetuneRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
