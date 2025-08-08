# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass, KW_ONLY
from typing import Any, Optional, Union

import torch
from omegaconf import DictConfig
from torch.distributed.checkpoint.state_dict import _init_optim_state, set_state_dict
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtune import config, training, utils
from torchtune.modules.peft import get_adapter_state_dict, get_merged_lora_ckpt
from torchtune.training.checkpointing import DistributedCheckpointer
from torchtune.training.checkpointing._checkpoint_client import (
    CheckpointClient,
    TrainingProgress,
)
from torchtune.training.memory import OptimizerInBackwardWrapper

from .._distributed import gather_cpu_state_dict

from ._checkpointer import DistributedStepCheckpointer

from ._utils import STEP_KEY

log = utils.get_logger("DEBUG")


class SDCheckpointClient(CheckpointClient):
    """
    Stateful checkpointing client for TorchTune recipes. This class is responsible for
    saving and loading checkpoints using the user configured checkpointers or distributed
    checkpointer if asynchronous checkpointing is enabled.

    Args:
        cfg (DictConfig): Configuration object used to instantiate the recipe.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__(cfg=cfg)

    def _get_checkpointer(self):
        """
        Builds and returns the user configured Checkpointer.
        """
        if not self._checkpointer:
            should_load_recipe_state: bool = (
                False
                if self._enable_async_checkpointing
                else self._resume_from_checkpoint
            )
            self._checkpointer = config.instantiate(
                self._cfg.checkpointer,
                should_load_recipe_state=should_load_recipe_state,
                use_grouped_gemm=self._cfg.model.get("use_grouped_gemm", True),
            )
        return self._checkpointer

    def _save_checkpoint_sync(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OptimizerInBackwardWrapper],
        training_progress: TrainingProgress,
        epoch: int,
        adapter_config: Optional[dict[str, Any]],
        adapter_only: bool,
        single_device: bool,
    ) -> None:
        """
        Checkpoint the training state synchronously.
        The constructed checkpoint state dict contains the following information:
        - Model weights with key training.MODEL_KEY
        - Relevant recipe state, including optimizer, if training is not complete

        To correctly resume training from this checkpoint, user needs to have both
        resume_from_checkpoint flag set to True and recipe file paths set in the config.
        """
        intermediate_checkpoint = epoch + 1 < training_progress.total_epochs
        checkpointer = self._get_checkpointer()
        no_dist = not isinstance(checkpointer, DistributedCheckpointer)

        # final dict passed onto the checkpointer
        checkpoint_dict = {}

        if self._is_rank_zero:
            log.info(
                "Saving checkpoint. This may take some time. Retrieving full model state dict..."
            )
            cp_start = time.perf_counter()

        model_state_dict = {}
        optim_state_dict = {}

        if no_dist and not single_device:
            # To prevent GPU memory from spiking during checkpoint save,
            # we consolidate the full model and optim state dicts on CPU for rank 0
            model_state_dict = gather_cpu_state_dict(
                model,
                self._is_rank_zero,
                device=self._device,
            )

            if self._is_rank_zero:
                log.info(
                    f"Getting full model state dict took {time.perf_counter() - cp_start:.2f} secs"
                )
        elif no_dist:
            model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            model_state_dict = model.state_dict()

        if intermediate_checkpoint:
            if self._is_rank_zero:
                log.info("Getting optimizer state dict...")
                optim_start = time.perf_counter()

            if no_dist:
                if not self._optimizer_in_bwd:
                    optim_state_dict = training.get_full_optimizer_state_dict(
                        model,
                        optimizer,
                        self._is_rank_zero,
                        device=self._device,
                    )
                else:
                    for param, opt in optimizer.optim_map.items():
                        optim_state_dict[
                            param
                        ] = training.get_full_optimizer_state_dict(
                            model, opt, self._is_rank_zero, device=self._device
                        )
            else:
                optim_state_dict = optimizer.state_dict()

            if self._is_rank_zero:
                log.info(
                    f"Getting optimizer state dict took {time.perf_counter() - optim_start:.2f} secs"
                )
        else:
            optim_state_dict = None

        def _save_checkpoint_helper():
            # if training is in-progress, checkpoint the optimizer state and recipe state
            # as well.
            if intermediate_checkpoint:
                checkpoint_dict.update({training.OPT_KEY: optim_state_dict})
                checkpoint_dict.update(training_progress.state_dict())

            if adapter_config is not None:
                checkpoint_dict.update(
                    {
                        training.ADAPTER_KEY: get_adapter_state_dict(model_state_dict),
                        training.ADAPTER_CONFIG: adapter_config,
                    }
                )

                get_merged_lora_ckpt(
                    model_state_dict, adapter_config["r"], adapter_config["lora_alpha"]
                )

            checkpoint_dict.update(
                {
                    training.MODEL_KEY: model_state_dict,
                }
            )

            self._get_checkpointer().save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
                adapter_only=adapter_only,
            )

            if self._is_rank_zero:
                log.info(
                    f"Saving checkpoint took {time.perf_counter() - cp_start:.2f} secs"
                )

        # Now that we have the model and optim state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if no_dist and not single_device:
            if self._is_rank_zero:
                _save_checkpoint_helper()

            torch.distributed.barrier()
        else:
            _save_checkpoint_helper()


@dataclass
class StepTrainingProgress(TrainingProgress):
    """
    This is training progress metadata for step-based checkpointing.
    """

    _: KW_ONLY
    global_step: int

    def state_dict(self) -> dict[str, object]:
        return {
            **super().state_dict(),
            STEP_KEY: self.global_step,
        }


# Inspired by torchtune.training.checkpointing._checkpoint_client.CheckpointClient
class StepCheckpointClient:
    """
    Checkpoint client for step-based checkpointing.
    """

    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir

        self._dcp_checkpointer = None

        self._world_size, self._rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self._rank == 0

    def load_latest_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | OptimizerInBackwardWrapper,
        dataloader: StatefulDataLoader,
    ) -> dict[str, Any] | None:
        """
        This method is used to resume training from a distributed step-basedcheckpoint state.
        Due to being distributed, this method is called on every rank.
        """

        if self._is_rank_zero:
            dcp_load_start = time.perf_counter()

        _init_optim_state(optimizer)

        # Build the state dict to be loaded from the distributed checkpoint
        checkpoint_dict: dict[str, Any] = {}
        model_state_dict = model.state_dict()
        optim_state_dict = optimizer.state_dict()
        dataloader_state_dict = dataloader.state_dict()

        # Hack to properly initialize the learning rate scheduler
        # TODO: Find a better way to do this, possibly by including the following
        # code in _init_optim_state
        if "param_groups" in optim_state_dict:
            for param_group in optim_state_dict["param_groups"]:
                if param_group.get("initial_lr") is None:
                    param_group[
                        "initial_lr"
                    ] = 0.0  # This will get overriden by the actual value in optimizer

        checkpoint_dict.update(
            {
                training.MODEL_KEY: model_state_dict,
                training.OPT_KEY: optim_state_dict,
                **StepTrainingProgress(
                    seed=0,
                    epochs_run=0,
                    total_epochs=0,
                    max_steps_per_epoch=0,
                    dataloader_state_dict=dataloader_state_dict,
                    global_step=0,
                ).state_dict(),
            }
        )

        for rank in range(self._world_size):
            if rank == self._rank:
                checkpoint_dict[
                    f"{training.DATALOADER_KEY}-{rank}"
                ] = checkpoint_dict.pop(training.DATALOADER_KEY)
            else:
                # According to the comments in torch.distributed.checkpoint.state_dict_loader.load,
                # the state_dict should have the same keys on all ranks. Mismatched keys may result in hangs or errors.
                # We ensure this requirement by creating placeholder entries here
                # even though such issues have not been observed without this.
                # (Why? My best guess is that each rank's dataloader state is only saved and loaded by the rank itself,
                # thus eliminating collective communication concerns.)
                # These placeholder keys need non-strict load (see DefaultLoadPlanner(allow_partial_load=True)
                # in .checkpointer.DistributedStepCheckpointer.load_checkpoint)
                # since the state dict on each rank passed to async_save does not contain the keys of other ranks,
                # which is presumably recorded in the metadata file.
                checkpoint_dict[f"{training.DATALOADER_KEY}-{rank}"] = None

        latest_checkpoint_path = (
            self._get_dcp_checkpointer().get_latest_checkpoint_path()
        )
        if latest_checkpoint_path is None:
            return None
        self._get_dcp_checkpointer().load_checkpoint(
            state_dict=checkpoint_dict,
            checkpoint_path=latest_checkpoint_path,
        )

        # Ensure load post-processing although the state dict has already been modifed in place
        set_state_dict(
            model,
            optimizer,
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
            optim_state_dict=checkpoint_dict[training.OPT_KEY],
        )
        dataloader.load_state_dict(
            checkpoint_dict[f"{training.DATALOADER_KEY}-{self._rank}"]
        )

        if self._is_rank_zero:
            log.info(
                f"Checkpoint loaded in {time.perf_counter() - dcp_load_start:.2f} seconds",
            )

        return checkpoint_dict

    def save_checkpoint_async(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | OptimizerInBackwardWrapper,
        training_progress: StepTrainingProgress,
    ) -> None:
        """
        Checkpoint the training state asynchronously as a distributed checkpoint. Saving
        asnchronously unblocks the training sooner to continue for the next step.
        The constructed checkpoint state dict contains the following information:
        - Model weights with key training.MODEL_KEY
        - Relevant recipe state, including optimizer, if training is not complete

        User does not need to provide any paths to checkpoint or recipe files. Latest intermediate
        and valid checkpoint will be loaded from the output directory and training progress will be
        restored automatically.
        """

        if self._is_rank_zero:
            log.info("Saving checkpoint asynchronously. Retrieving full state dict...")
            cp_start = time.perf_counter()

        # Create the checkpoint dict to be sent to the checkpointer and ultimately persisted to storage
        ckpt_dict = {}
        ckpt_dict.update(training_progress.state_dict())

        # Rename the key of the dataloader state to include the rank
        # so that the key is unique across ranks and avoids being deduplicated
        # in torch.distributed.checkpoint._dedup_save_plans.dedup_save_plans
        ckpt_dict[f"{training.DATALOADER_KEY}-{self._rank}"] = ckpt_dict.pop(
            training.DATALOADER_KEY
        )

        ckpt_dict[training.MODEL_KEY] = model.state_dict()
        ckpt_dict[training.OPT_KEY] = optimizer.state_dict()

        dcp_saver = self._get_dcp_checkpointer()
        dcp_saver.save_checkpoint_async(
            ckpt_dict,
            epoch=training_progress.epochs_run,
            step=training_progress.global_step,
            should_delete_stale_step_checkpoints=True,
        )

        if self._is_rank_zero:
            log.info(
                f"Checkpoint saved in {time.perf_counter() - cp_start:.2f} seconds",
            )

    def _get_dcp_checkpointer(self) -> DistributedStepCheckpointer:
        """
        Builds and returns the DistributedStepCheckpointer.
        DistributedStepCheckpointer is used for asynchronous step-based checkpointing.
        Uses the user configured output directory.
        """
        if not self._dcp_checkpointer:
            self._dcp_checkpointer = DistributedStepCheckpointer(
                output_dir=self._output_dir,
            )

        return self._dcp_checkpointer
