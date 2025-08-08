# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import gc
import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
from fsspec.core import url_to_fs
from safetensors.torch import save as save_safetensors, save_file
from torch.distributed.checkpoint import (
    async_save,
    DefaultLoadPlanner,
    FileSystemReader,
    FileSystemWriter,
    load,
    save,
)
from torchtune import training
from torchtune.models import convert_weights
from torchtune.training.checkpointing._checkpointer import _CheckpointerInterface
from torchtune.training.checkpointing._utils import (
    ADAPTER_CONFIG_FNAME,
    ADAPTER_MODEL_FNAME,
    check_outdir_not_in_ckptdir,
    copy_files,
    get_adapter_checkpoint_path,
    get_all_checkpoints_in_dir,
    get_model_checkpoint_path,
    get_recipe_checkpoint_path,
    RECIPE_STATE_DIRNAME,
    REPO_ID_FNAME,
    safe_torch_load,
    SAFETENSOR_INDEX_FNAME,
    SHARD_FNAME,
    SUFFIXES_TO_NOT_COPY,
    TORCH_INDEX_FNAME,
)
from torchtune.utils import get_logger, get_world_size_and_rank, log_rank_zero

from ._utils import ModelType

__targets__ = ("torchtune.training.checkpointing._checkpointer", "torchtune.training")
__implements__ = ("FullModelHFCheckpointer",)

logger = get_logger("DEBUG")


# Adapted from
# https://github.com/pytorch/torchtune/blob/337cd7c53d7006e2330b2f0b248d48ec5180b6cc/torchtune/training/checkpointing/_checkpointer.py
# Differences from torchtune.training.checkpointing._checkpointer.FullModelHFCheckpointer:
#   - Add support for Qwen3 model type
class FullModelHFCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in HF's format. For LoRA models this includes
    saving checkpoints in a format that can be loaded into PEFT via e.g. ``from_pretrained``. Examples include
    the Llama-2-7b-hf model from the meta-llama repo (https://huggingface.co/meta-llama/Llama-2-7b-hf).

    Note:
        HF checkpoint names are usually ordered by ID (eg: 0001_of_0003, 0002_of_0003, etc.) To ensure \
        we read the files in the right order, we sort the checkpoint file names before reading.

    Note:
        Checkpoint conversion to and from HF's format requires access to model params which are \
        read directly from the ``config.json`` file. This helps ensure we either load the weights \
        correctly or error out in case of discrepancy between the HF checkpoint file and torchtune's \
        model implementations.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (Union[list[str], dict[str, str]]): list of checkpoint files to load or a dictionary
            containing the keys keys ["filename_format", "max_filename"]. Since the checkpointer takes care
            of sorting by file ID, the order in this list does not matter.
        model_type (str): Model type of the model for which the checkpointer is being loaded, e.g. LLAMA3.
        use_grouped_gemm (bool): Whether the moe model's experts use grouped gemm or not. True is suggested.
        output_dir (Optional[str]): Directory to save the checkpoint files, default None.
        adapter_checkpoint (Optional[str]): Path to the adapter weights. If None,
            and `should_load_recipe_state=True`, then look for adapter_model.pt in output_dir/epoch_{largest_epoch}.
            Default is None.
        recipe_checkpoint (Optional[str]): Path to the recipe state checkpoint file. If None,
            and `should_load_recipe_state=True`, then look for recipe_state.pt in output_dir/RECIPE_STATE_DIRNAME.
            Default is None.
        resume_from_checkpoint (bool): If True, the checkpointer will load the additional checkpoint files corresponding to
            the receipe state from a previous run. Default is False. This flag is deprecated. Please use
            the should_load_recipe_state flag instead.
        safe_serialization (bool): If True, the checkpointer will save the checkpoint file using `safetensors`.
            Default is True.
        should_load_recipe_state (bool): If True, the checkpointer will load the additional checkpoint files corresponding to
            the receipe state from a previous run. Default is False
        enable_dcp (bool): If True, the checkpointer will load the checkpoint file using dcp checkpointing apis.
            This is currently an experimental feature.

    Raises:
        ValueError: If ther checkpoint_dir and output_dir are not on the same filesystem
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: Union[list[str], dict[str, str]],
        model_type: str,
        use_grouped_gemm: bool = True,
        output_dir: Optional[str] = None,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
        safe_serialization: bool = True,
        should_load_recipe_state: bool = False,
        enable_dcp: bool = False,
    ) -> None:
        self._should_load_recipe_state = should_load_recipe_state
        if resume_from_checkpoint:
            self._should_load_recipe_state = resume_from_checkpoint
            logger.warning(
                "*resume_from_checkpoint is deprecated. Please use the 'should_load_recipe_state' instead"
            )

        self._safe_serialization = safe_serialization
        self._checkpoint_dir = checkpoint_dir
        self._model_type = ModelType[model_type]
        self._use_grouped_gemm = use_grouped_gemm
        self._enable_dcp = enable_dcp
        self._fs, _ = url_to_fs(self._checkpoint_dir)
        self._output_dir = output_dir

        if self._output_dir is not None:
            check_outdir_not_in_ckptdir(
                ckpt_dir=self._checkpoint_dir, out_dir=self._output_dir
            )
            output_fs, _ = url_to_fs(self._output_dir)
            if self._fs != output_fs:
                raise ValueError(
                    f"Checkpoint and output directories must be on the same filesystem. "
                    f"Got {self._fs} and {output_fs} instead."
                )
            self._fs.mkdirs(output_dir, exist_ok=True)

        # weight_map contains the state_dict key -> checkpoint file mapping so we can correctly
        # parition the state dict into output checkpoint files. This is updated during checkpoint
        # load
        self._weight_map: dict[str, str] = None

        # the config.json file contains model params needed for state dict conversion
        self._config = None
        with self._fs.open(
            os.path.join(self._checkpoint_dir, "config.json"), "r"
        ) as json_file:
            self._config = json.loads(json_file.read())

        # repo_id is necessary for when saving an adapter config, so its compatible with HF.
        # This json file is produced and saved in the download step.
        # contents are {"repo_id": "some_model/some_model_version"}
        repo_id_path = os.path.join(self._checkpoint_dir, REPO_ID_FNAME) + ".json"

        self.repo_id = None
        if self._fs.exists(repo_id_path):
            with self._fs.open(repo_id_path, "r") as json_file:
                data = json.load(json_file)
                self.repo_id = data.get("repo_id")

        #  resume from adapter_model ckpt
        self._adapter_checkpoint = get_adapter_checkpoint_path(
            output_dir=self._output_dir,
            adapter_checkpoint=adapter_checkpoint,
            should_load_recipe_state=self._should_load_recipe_state,
            pattern=r"^epoch_(\d+)",
        )

        # resume recipe_state ckpt
        self._recipe_checkpoint = get_recipe_checkpoint_path(
            output_dir=self._output_dir,
            recipe_checkpoint=recipe_checkpoint,
            should_load_recipe_state=self._should_load_recipe_state,
        )

        # get ckpt paths
        self._checkpoint_paths = get_model_checkpoint_path(
            checkpoint_files=checkpoint_files,
            checkpoint_dir=self._checkpoint_dir,
            output_dir=self._output_dir,
            should_load_recipe_state=self._should_load_recipe_state,
            has_adapter_checkpoint=self._adapter_checkpoint is not None,
        )

        if self._should_load_recipe_state:
            logger.info(
                "Loading the recipe state using: "
                f"\n\tcheckpoint_paths: {[str(path) for path in self._checkpoint_paths]}"
                f"\n\trecipe_checkpoint: {self._recipe_checkpoint}"
                f"\n\tadapter_checkpoint: {self._adapter_checkpoint}"
            )

    def load_checkpoint(self) -> dict[str, Any]:
        """
        Load HF checkpoint from file.

        The keys and weights from across all checkpoint files are merged into a single state_dict.
        We preserve the "state_dict key" <-> "checkpoint file" mapping in weight_map so we can
        write the state dict correctly in ``save_checkpoint``.

        Before returning, the model state dict is converted to a torchtune-compatible format using
        the appropriate convert_weights function (depending on ``self._model_type``).

        Returns:
            state_dict (dict[str, Any]): torchtune checkpoint state dict

        Raises:
            ValueError: If the values in the input state_dict are not Tensors
        """

        self._weight_map = {}

        # merged state_dict contains keys and weights from all the checkpoint files
        merged_state_dict: dict[str, torch.Tensor] = {}

        # converted_state_dict is the final state_dict passed to the recipe after the
        # keys are converted into the torchtune format. This optionally also contains
        # the recipe state and adapter weights
        converted_state_dict: dict[str, dict[str, torch.Tensor]] = {}

        if self._enable_dcp:
            from torch.distributed.checkpoint import (
                _HuggingFaceLoadPlanner,
                _HuggingFaceStorageReader,
            )

            # DCP load using the storage reader
            hf_storage_reader = _HuggingFaceStorageReader(path=self._checkpoint_dir)
            metadata = hf_storage_reader.read_metadata()
            state_dict = {}
            for key in metadata.state_dict_metadata.keys():
                # arbitrary value to ensure that the state_dict is not empty
                state_dict[key] = torch.empty(1)

            self._weight_map = metadata.storage_data

            load(
                state_dict=state_dict,
                storage_reader=hf_storage_reader,
                planner=_HuggingFaceLoadPlanner(allow_tensor_resize=True),
            )

            merged_state_dict = state_dict
        else:
            # _checkpoint_paths are already sorted so simply enumerate to generate the right id
            for cpt_idx, cpt_path in enumerate(self._checkpoint_paths):
                state_dict = safe_torch_load(cpt_path)
                for key, value in state_dict.items():
                    # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
                    # will break recipe code
                    if not isinstance(value, torch.Tensor):
                        raise ValueError(
                            f"Expected all values in the state dict to be torch.Tensor. "
                            f"Found {type(value)} instead."
                        )
                    # idx is written in the 4 digit format (eg: 0001, 0002, etc.)
                    self._weight_map[key] = f"{cpt_idx + 1:04}"
                merged_state_dict.update(state_dict)

                # delete the state_dict to free up memory; TODO check if this del is needed
                del state_dict
                gc.collect()

        if self._model_type in (ModelType.PHI3_MINI, ModelType.PHI4):
            log_rank_zero(
                logger=logger,
                msg="Converting Phi weights from HF format."
                "Note that conversion of adapter weights into PEFT format is not supported.",
            )
            from torchtune.models.phi3._convert_weights import phi3_hf_to_tune

            num_heads = self._config["num_attention_heads"]
            num_kv_heads = self._config["num_key_value_heads"]
            dim = self._config["hidden_size"]

            # Should only pass num_heads, num_kv_heads, dim for GQA
            if num_heads == num_kv_heads:
                num_heads, num_kv_heads, dim = None, None, None

            converted_state_dict[training.MODEL_KEY] = phi3_hf_to_tune(
                merged_state_dict,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dim=dim,
            )
        elif self._model_type == ModelType.REWARD:
            from torchtune.rlhf.utils import reward_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = reward_hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
            )
        elif self._model_type == ModelType.QWEN2:
            from torchtune.models.qwen2._convert_weights import qwen2_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = qwen2_hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                tie_word_embeddings=self._config["tie_word_embeddings"],
            )
        elif self._model_type == ModelType.QWEN3:
            from mirotrain.models.qwen3 import qwen3_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = qwen3_hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                tie_word_embeddings=self._config["tie_word_embeddings"],
            )
        elif self._model_type == ModelType.QWEN3_MoE:
            from mirotrain.models.qwen3_moe._convert_weights import qwen3_moe_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = qwen3_moe_hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                tie_word_embeddings=self._config["tie_word_embeddings"],
                num_layers=self._config["num_hidden_layers"],
                num_experts=self._config["num_experts"],
                intermediate_dim=self._config["moe_intermediate_size"],
                use_grouped_gemm=self._use_grouped_gemm,
            )
        elif self._model_type == ModelType.LLAMA3_VISION:
            from torchtune.models.llama3_2_vision._convert_weights import (
                llama3_vision_hf_to_tune,
            )

            text_config = self._config.get("text_config", {})
            vision_config = self._config.get("vision_config", {})
            converted_state_dict[training.MODEL_KEY] = llama3_vision_hf_to_tune(
                merged_state_dict,
                num_heads=text_config["num_attention_heads"],
                num_kv_heads=text_config["num_key_value_heads"],
                dim=text_config["hidden_size"],
                head_dim=text_config.get("head_dim", None),
                vocab_size=text_config["vocab_size"],
                cross_attention_layers=text_config.get("cross_attention_layers", None),
                encoder_dim=vision_config["hidden_size"],
                tile_size=vision_config["image_size"],
                num_tiles=vision_config["max_num_tiles"],
                supported_aspect_ratios=vision_config.get(
                    "supported_aspect_ratios", None
                ),
            )
        elif self._model_type == ModelType.CLIP_TEXT:
            from torchtune.models.clip._convert_weights import clip_text_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = clip_text_hf_to_tune(
                merged_state_dict,
            )
        elif self._model_type == ModelType.GEMMA2:
            from torchtune.models.gemma2._convert_weights import gemma2_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = gemma2_hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                head_dim=self._config.get("head_dim", None),
            )
        elif self._model_type == ModelType.T5_ENCODER:
            from torchtune.models.t5._convert_weights import t5_encoder_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = t5_encoder_hf_to_tune(
                merged_state_dict,
            )
        elif self._model_type == ModelType.LLAMA4:
            from torchtune.models.llama4._convert_weights import llama4_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = llama4_hf_to_tune(
                merged_state_dict,
            )
        else:
            converted_state_dict[training.MODEL_KEY] = convert_weights.hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                head_dim=self._config.get("head_dim", None),
            )

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            converted_state_dict[training.ADAPTER_KEY] = adapter_state_dict

        if self._should_load_recipe_state:
            recipe_state = safe_torch_load(self._recipe_checkpoint, mmap=False)
            converted_state_dict.update(recipe_state)

        return converted_state_dict

    def save_checkpoint(
        self,
        state_dict: dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
        *,
        step: Optional[int] = None,
    ) -> None:
        """
        Save HF checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir/RECIPE_STATE_DIRNAME``
        which contains the recipe state.

        The state_dict is first converted back to the HF format and then partitioned based on the
        ``_weight_map`` into separate checkpoint files.

        Args:
            state_dict (dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            intermediate_checkpoint (bool): If True, an additional checkpoint files for recipe state
                and (if applicable) adapter weights are created. Default is False
            adapter_only (bool): If True, only save the adapter weights. Default is False
            step (Optional[int]): Step number. Used to create the checkpoint file name.

        Raises:
            ValueError: if ``adapter_only`` is True and adapter checkpoint not found in state_dict.
        """
        if self._output_dir is None:
            raise ValueError(
                "Output directory not specified. Please specify an output directory to save the checkpoint."
            )
        output_dirname = f"step_{step}" if step is not None else f"epoch_{epoch}"
        # convert the state_dict back to hf format; do this inplace
        if not adapter_only:
            if self._model_type in (ModelType.PHI3_MINI, ModelType.PHI4):
                from torchtune.models.phi3._convert_weights import phi3_tune_to_hf

                state_dict[training.MODEL_KEY] = phi3_tune_to_hf(
                    state_dict[training.MODEL_KEY]
                )
            elif self._model_type == ModelType.REWARD:
                from torchtune.rlhf.utils import reward_tune_to_hf

                state_dict[training.MODEL_KEY] = reward_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                )
            elif self._model_type == ModelType.QWEN2:
                from torchtune.models.qwen2._convert_weights import qwen2_tune_to_hf

                state_dict[training.MODEL_KEY] = qwen2_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    tie_word_embeddings=self._config["tie_word_embeddings"],
                )
            elif self._model_type == ModelType.QWEN3:
                from mirotrain.models.qwen3 import qwen3_tune_to_hf

                state_dict[training.MODEL_KEY] = qwen3_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    tie_word_embeddings=self._config["tie_word_embeddings"],
                )
            elif self._model_type == ModelType.QWEN3_MoE:
                from mirotrain.models.qwen3_moe._convert_weights import (
                    qwen3_moe_tune_to_hf,
                )

                state_dict[training.MODEL_KEY] = qwen3_moe_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    tie_word_embeddings=self._config["tie_word_embeddings"],
                    num_layers=self._config["num_hidden_layers"],
                    num_experts=self._config["num_experts"],
                    intermediate_dim=self._config["moe_intermediate_size"],
                    use_grouped_gemm=self._use_grouped_gemm,
                )
            elif self._model_type == ModelType.LLAMA3_VISION:
                from torchtune.models.llama3_2_vision._convert_weights import (
                    llama3_vision_tune_to_hf,
                )

                text_config = self._config.get("text_config", {})
                vision_config = self._config.get("vision_config", {})
                state_dict[training.MODEL_KEY] = llama3_vision_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=text_config["num_attention_heads"],
                    num_kv_heads=text_config["num_key_value_heads"],
                    dim=text_config["hidden_size"],
                    head_dim=text_config.get("head_dim", None),
                    vocab_size=text_config["vocab_size"],
                    cross_attention_layers=text_config.get(
                        "cross_attention_layers", None
                    ),
                    encoder_dim=vision_config["hidden_size"],
                    tile_size=vision_config["image_size"],
                    num_tiles=vision_config["max_num_tiles"],
                    supported_aspect_ratios=vision_config.get(
                        "supported_aspect_ratios", None
                    ),
                )
            elif self._model_type == ModelType.GEMMA2:
                from torchtune.models.gemma2._convert_weights import gemma2_tune_to_hf

                state_dict[training.MODEL_KEY] = gemma2_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )
            elif self._model_type == ModelType.LLAMA4:
                from torchtune.models.llama4._convert_weights import llama4_tune_to_hf

                state_dict[training.MODEL_KEY] = llama4_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                )
            else:
                state_dict[training.MODEL_KEY] = convert_weights.tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )

            if self._enable_dcp:
                from torch.distributed.checkpoint import (
                    _HuggingFaceSavePlanner,
                    _HuggingFaceStorageWriter,
                )

                # DCP save using the storage writer
                fqn_to_file_index_mapping = {}
                for fqn, filename in self._weight_map.items():
                    index = int(filename.split("-")[1])
                    fqn_to_file_index_mapping[fqn] = index
                storage_writer = _HuggingFaceStorageWriter(
                    path=os.path.join(self._output_dir, f"epoch_{epoch}"),
                    fqn_to_index_mapping=fqn_to_file_index_mapping,
                )
                save(
                    state_dict=state_dict[training.MODEL_KEY],
                    storage_writer=storage_writer,
                    planner=_HuggingFaceSavePlanner(),
                    no_dist=True,
                )
            else:
                # split the state_dict into separate dicts, one for each output checkpoint file
                # e.g. split_state_dicts= {
                #       "0001": {"key1": tensor1, "key2": tensor2},
                #       "0002": {"key3": tensor3}
                #       }
                split_state_dicts: dict[str, dict[str, torch.Tensor]] = {}
                total_size = 0
                for key, weight in state_dict[training.MODEL_KEY].items():
                    cpt_idx = self._weight_map[key]

                    # initialize dict
                    if cpt_idx not in split_state_dicts:
                        split_state_dicts[cpt_idx] = {}

                    split_state_dicts[cpt_idx].update({key: weight})
                    total_size += weight.numel() * weight.element_size()

                # write the partitioned state dicts to the right checkpoint file
                # e.g. model-00001-of-00004.safetensors, model-00002-of-00004.safetensors, etc
                num_shards = len(split_state_dicts)
                map_original_name_to_new_name = {}
                for cpt_idx, model_state_dict in split_state_dicts.items():
                    # TODO: We should probably use the original shard name and just add a prefix
                    # however, having the SHARD_FNAME standardizes our checkpoints
                    shard_name = SHARD_FNAME.format(
                        cpt_idx=f"{cpt_idx}".zfill(5),
                        num_shards=f"{num_shards}".zfill(5),
                    )
                    map_original_name_to_new_name[cpt_idx] = shard_name
                    output_path = os.path.join(
                        self._output_dir, output_dirname, shard_name
                    )
                    self._fs.mkdirs(os.path.dirname(output_path), exist_ok=True)

                    if not self._safe_serialization:
                        output_path = output_path = ".bin"
                        torch.save(model_state_dict, output_path)
                    else:
                        output_path = output_path + ".safetensors"
                        save_file(
                            model_state_dict, output_path, metadata={"format": "pt"}
                        )

                    logger.info(
                        "Model checkpoint of size "
                        f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                        f"saved to {output_path}"
                    )

                # Save the appropriate index file based on serialization format
                # e.g. {metadata: {total_size: 1234},
                # weight_map: {"key1": "model_0001.safetensors", "key2": "model_0002.safetensors"}}
                if self._safe_serialization:
                    weight_map = {
                        k: map_original_name_to_new_name[cpt_idx] + ".safetensors"
                        for k, cpt_idx in self._weight_map.items()
                    }
                    index_file_name = SAFETENSOR_INDEX_FNAME
                else:
                    weight_map = {
                        k: map_original_name_to_new_name[cpt_idx] + ".bin"
                        for k, cpt_idx in self._weight_map.items()
                    }
                    index_file_name = TORCH_INDEX_FNAME

                index_path = os.path.join(
                    self._output_dir, output_dirname, index_file_name
                )

                index_data = {
                    "metadata": {"total_size": total_size},
                    "weight_map": weight_map,
                }
                with self._fs.open(index_path, "w") as f:
                    json.dump(index_data, f, indent=2)

        if training.ADAPTER_KEY in state_dict:
            # TODO: saving it "as is" is a requirement because, if we only save with
            # convert_weights.tune_to_peft_adapter_weights, we do NOT have a fn
            # convert_weights.peft_to_tune. The .pt format is not needed, but
            # it is an easy way to distinguish the adapters. Ideally we should save only one.
            output_path = (
                os.path.join(self._output_dir, output_dirname, ADAPTER_MODEL_FNAME)
                + ".pt"
            )
            self._fs.mkdirs(os.path.dirname(output_path), exist_ok=True)
            with self._fs.open(output_path, "wb") as f:
                torch.save(state_dict[training.ADAPTER_KEY], f)
            logger.info(
                "Adapter checkpoint of size "
                f"{self._fs.size(output_path) / 1024**3:.2f} GiB "
                f"saved to {output_path}"
            )

            if self._model_type in (ModelType.PHI3_MINI, ModelType.PHI4):
                logger.warning(
                    "Saving Phi adapter weights to PEFT format is not supported, saving to torchtune format instead"
                )
            elif self._model_type == ModelType.LLAMA3_VISION:
                logger.warning(
                    "Saving Llama3.2 Vision adapter weights to PEFT format is not supported, saving to torchtune format instead"
                )
            elif self._model_type == ModelType.LLAMA4:
                logger.warning(
                    "Saving Llama4 adapter weights to PEFT format is not supported, saving to torchtune format instead"
                )
            else:
                config = (
                    self._config["text_config"]
                    if "text_config" in self._config
                    else self._config
                )
                state_dict[
                    training.ADAPTER_KEY
                ] = convert_weights.tune_to_peft_adapter_weights(
                    state_dict[training.ADAPTER_KEY],
                    num_heads=config["num_attention_heads"],
                    num_kv_heads=config["num_key_value_heads"],
                    dim=config["hidden_size"],
                    head_dim=config.get("head_dim", None),
                )
                output_path = os.path.join(
                    self._output_dir, output_dirname, ADAPTER_MODEL_FNAME
                )
                self._fs.mkdirs(os.path.dirname(output_path), exist_ok=True)
                if not self._safe_serialization:
                    output_path = output_path + ".bin"
                    with self._fs.open(output_path, "wb") as f:
                        torch.save(state_dict[training.ADAPTER_KEY], f)
                else:
                    output_path = output_path + ".safetensors"
                    with self._fs.open(output_path, "wb") as f:
                        save_bytes = save_safetensors(
                            state_dict[training.ADAPTER_KEY],
                            metadata={"format": "pt"},
                        )
                        f.write(save_bytes)
                logger.info(
                    "Adapter checkpoint of size "
                    f"{self._fs.size(output_path) / 1024**3:.2f} GiB "
                    f"saved to {output_path}"
                )
        elif adapter_only:
            raise ValueError(
                "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
            )

        if training.ADAPTER_CONFIG in state_dict:
            if self._model_type in (ModelType.PHI3_MINI, ModelType.PHI4):
                logger.warning(
                    "PEFT integration for Phi is not supported, skipping adapter config save"
                )
            elif self._model_type == ModelType.LLAMA3_VISION:
                logger.warning(
                    "PEFT integration for Llama3.2 Vision is not supported, skipping adapter config save"
                )
            else:
                state_dict[
                    training.ADAPTER_CONFIG
                ] = convert_weights.tune_to_peft_adapter_config(
                    adapter_config=state_dict[training.ADAPTER_CONFIG],
                    base_model_name_or_path=self.repo_id,
                )
                output_path = (
                    os.path.join(self._output_dir, output_dirname, ADAPTER_CONFIG_FNAME)
                    + ".json"
                )
                with self._fs.open(output_path, "w") as f:
                    json.dump(state_dict[training.ADAPTER_CONFIG], f)
                logger.info(
                    "Adapter checkpoint of size "
                    f"{self._fs.size(output_path) / 1024**3:.2f} GiB "
                    f"saved to {output_path}"
                )

        # Save all files in ckpt_dir, except model weights and mapping, to output_dir/epoch_{epoch}
        # So its easy to run inference with the model using this epoch's checkpoint
        copy_files(
            self._checkpoint_dir,
            os.path.join(self._output_dir, output_dirname),
            ignore_suffixes=SUFFIXES_TO_NOT_COPY,
        )

        # If the recipe state needs to be output, first remove the model state dict
        # and if it exists, remove the adapter state dict as well
        if intermediate_checkpoint:
            _ = state_dict.pop(training.MODEL_KEY, None)
            _ = state_dict.pop(training.ADAPTER_KEY, None)
            _ = state_dict.pop(training.ADAPTER_CONFIG, None)
            output_path = os.path.join(
                self._output_dir, RECIPE_STATE_DIRNAME, "recipe_state.pt"
            )
            parent_path = os.path.dirname(output_path)
            self._fs.mkdirs(parent_path, exist_ok=True)
            torch.save(state_dict, output_path)
            logger.info(
                "Recipe checkpoint of size "
                f"{self._fs.size(output_path) / 1024**3:.2f} GiB "
                f"saved to {output_path}"
            )
        else:
            logger.info("Saving final epoch checkpoint.")
            if adapter_only:
                logger.info(
                    "Please note that you have set adapter_only=True, so only adapter weights will be saved."
                    "You need to merge the adapter weights into your base model for further use. "
                    f"See {self.__class__.__name__}.save_checkpoint for more details."
                )
            else:
                logger.info(
                    "The full model checkpoint, including all weights and configurations, has been saved successfully."
                    "You can now use this checkpoint for further training or inference."
                )


# Inspired by torchtune.training.checkpointing._checkpointer.DistributedCheckpointer
class DistributedStepCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in the DistributedCheckpointing format.

    Args:
        output_dir (str): Directory to save/load the checkpoint files
        process_group (Optional[dist.ProcessGroup]): Optional process group to use
            for distributed saving/loading. If None, the default process group will be used.
            For checkpointing, gloo CPU-based backend is needed.
    """

    def __init__(
        self,
        output_dir: str,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._process_group = process_group

        self.checkpoint_future: Future | None = None

        self._checkpoint_dir_epoch_template = "epoch_{epoch}"
        self._checkpoint_dir_step_template = "step_{step}"
        self._metadata_file = ".metadata"
        _, self._rank = get_world_size_and_rank()

    def get_latest_checkpoint_path(self) -> Path | None:
        sorted_step_checkpoint_paths = self._get_sorted_checkpoint_paths()
        if not sorted_step_checkpoint_paths:
            return None
        return sorted_step_checkpoint_paths[0]

    def load_checkpoint(
        self, state_dict: dict[str, Any], checkpoint_path: Path
    ) -> None:
        log_rank_zero(
            logger, msg=f"Loading step-based checkpoint from {checkpoint_path}"
        )
        load(
            state_dict=state_dict,
            storage_reader=FileSystemReader(checkpoint_path),
            process_group=self._process_group,
            planner=DefaultLoadPlanner(
                # This is used to work with the placeholder entries of the dataloader state dict.
                # See detailed explanations in .checkpoint_client.StepCheckpointClient.load_latest_checkpoint.
                allow_partial_load=True
            ),
        )

    def save_checkpoint_async(
        self,
        state_dict: dict[str, Any],
        epoch: int,
        step: int,
        should_delete_stale_step_checkpoints: bool,
    ) -> None:
        log_rank_zero(
            logger,
            msg=f"Starting to save a step-based checkpoint for epoch_{epoch} step_{step}",
        )

        checkpoint_path = Path.joinpath(
            self._output_dir,
            self._checkpoint_dir_epoch_template.format(epoch=epoch),
            self._checkpoint_dir_step_template.format(step=step),
        )

        if self.checkpoint_future and not self.checkpoint_future.done():
            # Previous checkpoint needs to finish before saving the next one.
            wait_start = time.perf_counter()

            logger.info(
                f"Rank {self._rank}: previous step-based checkpoint has not finished. "
                "Checkpointing frequency is too high. Waiting...",
            )

            self.checkpoint_future.result()

            logger.info(
                f"Rank {self._rank}: waited {time.perf_counter() - wait_start:.2f} seconds "
                "for previous step-based checkpoint to finish",
            )
            self.checkpoint_future = None

        cp_start = time.perf_counter()

        def callback(
            f: Future,
        ) -> None:
            if f.exception() is None:
                logger.info(
                    f"Rank {self._rank}: Step-based checkpoint is saved asynchronously to {checkpoint_path} successfully.",
                )
                if should_delete_stale_step_checkpoints and 0 == self._rank:
                    self._delete_stale_checkpoints()
            else:
                logger.error(
                    f"Rank {self._rank}: Step-based checkpoint failed to save asynchronously to {checkpoint_path} "
                    f"with the exception {f.exception()}"
                )

        self.checkpoint_future = async_save(
            state_dict=state_dict,
            storage_writer=FileSystemWriter(
                checkpoint_path,
                thread_count=16,
                single_file_per_rank=True,
                sync_files=True,
            ),
            process_group=self._process_group,
        )

        logger.info(
            f"Rank {self._rank}: Trainer was blocked for {time.perf_counter() - cp_start:.2f} seconds "
            "for step-based checkpointing to finish...",
        )

        self.checkpoint_future.add_done_callback(callback)

    def _delete_stale_checkpoints(self) -> None:
        """
        Delete stale checkpoints, keeping only the most recent one.
        The method will keep the latest checkpoint based on epoch and step numbers.
        """

        stale_step_checkpoint_paths = self._get_stale_checkpoint_paths()

        # Delete stale checkpoints
        for step_path in stale_step_checkpoint_paths:
            try:
                shutil.rmtree(step_path)
                logger.info(f"Deleted stale checkpoint: {step_path}")
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {step_path}: {e}")

        # Clean up empty epoch directories
        for step_path in stale_step_checkpoint_paths:
            epoch_path = step_path.parent
            if epoch_path.is_dir():
                # NOTE: This code block is currently unreachable in practice
                # because we always save one HF checkpoint per epoch,
                # ensuring that epoch directories are never empty and thus never deleted.
                try:
                    # Check if epoch directory is empty
                    if not any(epoch_path.iterdir()):
                        shutil.rmtree(epoch_path)
                        logger.info(f"Deleted empty epoch directory: {epoch_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete empty epoch directory {epoch_path}: {e}"
                    )

    def _get_stale_checkpoint_paths(self) -> list[Path]:
        sorted_step_checkpoint_paths = self._get_sorted_checkpoint_paths()
        return sorted_step_checkpoint_paths[1:]

    def _get_sorted_checkpoint_paths(self) -> list[Path]:
        checkpoint_dir_epoch_pattern = self._checkpoint_dir_epoch_template.format(
            epoch=r"(\d+)"
        )
        checkpoint_dir_step_pattern = self._checkpoint_dir_step_template.format(
            step=r"(\d+)"
        )

        # Get all epoch directories
        epoch_paths = get_all_checkpoints_in_dir(
            self._output_dir, pattern=f"^{checkpoint_dir_epoch_pattern}$"
        )

        @dataclass
        class CheckpointInfo:
            epoch: int
            step: int
            path: Path

        # Collect all checkpoint paths with their (epoch, step) for sorting
        found_checkpoints: list[CheckpointInfo] = []

        for epoch_path in epoch_paths:
            if epoch_path.is_dir():
                # Get all step directories within this epoch
                step_paths = get_all_checkpoints_in_dir(
                    epoch_path, pattern=f"^{checkpoint_dir_step_pattern}$"
                )
                for step_path in step_paths:
                    if step_path.is_dir():
                        if not os.path.isfile(
                            os.path.join(step_path, self._metadata_file)
                        ):
                            log_rank_zero(
                                logger,
                                f"Skipping checkpoint {step_path} because it is missing the metadata file",
                                level=logging.WARNING,
                            )
                            continue

                        found_checkpoints.append(
                            CheckpointInfo(
                                epoch=int(
                                    re.findall(
                                        checkpoint_dir_epoch_pattern, epoch_path.name
                                    )[0]
                                ),
                                step=int(
                                    re.findall(
                                        checkpoint_dir_step_pattern, step_path.name
                                    )[0]
                                ),
                                path=step_path,
                            )
                        )

        # Sort by epoch first, then by step (most recent first)
        found_checkpoints.sort(
            key=lambda ckpt_info: (ckpt_info.epoch, ckpt_info.step), reverse=True
        )
        return [ckpt_info.path for ckpt_info in found_checkpoints]
