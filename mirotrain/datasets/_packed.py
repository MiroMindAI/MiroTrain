# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import random
from typing import Dict, Iterator, List, Optional

import torch
from torch.nn import functional as F

from torch.utils.data import Dataset, IterableDataset
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX, PACK_TYPE
from tqdm import tqdm

__targets__ = (
    "torchtune.datasets",
    "torchtune.datasets._alpaca",
    "torchtune.datasets._chat",
    "torchtune.datasets._grammar",
    "torchtune.datasets._instruct",
    "torchtune.datasets._samsum",
    "torchtune.datasets._slimorca",
)
__implements__ = ("PackedDataset",)


class PackedDataset(Dataset):
    """
    A dataset that packs multiple tokenized samples into fixed-length sequences.

    This class is adapted from `torchtune.datasets.PackedDataset` with a key enhancement:
    support for lazy initialization, allowing modification of `max_seq_len` via
    an external `packed_length` attribute in trainer.

    Unlike standard datasets where `max_seq_len` typically refers to the maximum length of
    a single tokenized sample, here it defines the maximum length of a *packed sequence*—
    a single continuous sequence created by concatenating multiple samples, optionally
    separated by special tokens.

    Args:
        ds (Dataset): dataset to sample pack. This should return a dictionary with field
            "tokens" and "labels" containing the tokenized and label samples.
        max_seq_len (int): Maximum number of tokens to pack
        padding_idx (int): padding index for the tokenizer. Default is 0.
        max_packs (Optional[int]): Maximum number of packs. Default is None, which will create as many
            packs as possible.
        split_across_pack (bool): if the last sample in a pack does not fit in ``max_seq_len``,
            split the sample into the next pack, or move it entirely to the beginning of the next pack.
            For pre-training, typically this is set to True for general text completion. For
            fine-tuning, typically this is set to False to avoid truncating sentences in instruct
            tuning. Default is False.
    """

    def __init__(
        self,
        ds: Dataset,
        *,
        max_seq_len: int,
        padding_idx: int = 0,
        max_packs: Optional[int] = None,
        split_across_pack: bool = False,
    ) -> None:
        self.ds = ds
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.max_packs = max_packs
        self.split_across_pack = split_across_pack
        # Where final samples will be held
        self.packs: List[PACK_TYPE] = []
        self.previous_sample_boundary: int = 0
        # Delay actual packing until first access
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return
        self._pack()
        self._initialized = True

    def _pack(self) -> None:
        """Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
        then append the buffer to self.packs as a single "packed" sample. Continue
        until max_packs or end of dataset."""
        # Buffer to hold samples until they are long enough to be added to self.packs
        current_pack = {
            "tokens": [],
            "labels": [],
            "input_pos": [],
            "seq_lens": [],
        }

        # Only show progress bar on rank 0
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            pbar = tqdm(total=len(self.ds), desc="Packing dataset", dynamic_ncols=True)

        indices = list(range(len(self.ds)))
        rng = random.Random(1024)
        rng.shuffle(indices)

        for idx in indices:
            sample = self.ds[idx]
            tokens, labels = sample["tokens"], sample["labels"]

            # If the dataset outputs samples that are larger than the specified
            # max_seq_len and we're unable to split it, user needs to modify
            # one of the two parameters
            seq_len = len(tokens)
            if seq_len > self.max_seq_len and not self.split_across_pack:
                raise ValueError(
                    f"Dataset sample is too long ({seq_len} > {self.max_seq_len}). "
                    "Please set `split_across_pack=True` or increase `max_seq_len`."
                )

            # Update the current pack
            current_pack["tokens"] += tokens
            current_pack["labels"] += labels
            current_pack["input_pos"] += [x % self.max_seq_len for x in range(seq_len)]
            current_pack["seq_lens"] += [seq_len]

            # If the current pack is over the max_seq_len, add it to self.packs and
            # retain any truncated or bumped samples for next pack
            while (
                len(current_pack["tokens"]) > self.max_seq_len
                and not self._should_stop_packing()
            ):
                current_pack = self._split_and_add_pack(current_pack)

            if rank == 0:
                pbar.update()

            # Keep track of previous sample boundary
            self.previous_sample_boundary = len(current_pack["tokens"])

            if self._should_stop_packing():
                break

        # Handle the last pack if there's leftover and we haven't filled up the max packs
        if len(current_pack["tokens"]) > 0 and (
            self.max_packs is None or len(self.packs) < self.max_packs
        ):
            # No need to handle splitting at this point so we can just add the current pack
            self._add_pack(current_pack)

    def _should_stop_packing(self) -> bool:
        """If max packs is set, stop packing when we reach that number."""

        if self.max_packs is not None and len(self.packs) == self.max_packs:
            return True
        return False

    def _split_and_add_pack(self, current_pack: PACK_TYPE) -> PACK_TYPE:
        """Splits the current pack at the boundary, processes it, adds it to ``self.packs`` and
        returns the start of the next pack."""

        if self.split_across_pack:
            boundary = self.max_seq_len
            # The last elem in ``seq_lens`` ensures that ``sum(seq_lens) == self.max_seq_len``
            leftover_seq_len = self.max_seq_len - sum(current_pack["seq_lens"][:-1])
            seq_len_padding = [leftover_seq_len] if leftover_seq_len > 0 else []
        else:
            boundary = self.previous_sample_boundary
            # If we aren't splitting across packs, we leave out the last sample b/c
            # it will go into the next pack
            seq_len_padding = []

        pack = {
            "tokens": current_pack["tokens"][:boundary],
            "labels": current_pack["labels"][:boundary],
            "input_pos": current_pack["input_pos"][:boundary],
            "seq_lens": current_pack["seq_lens"][:-1] + seq_len_padding,
        }

        # Process and add the pack
        self._add_pack(pack)

        # Return the length of the first sample in next pack if we are splitting across packs,
        # otherwise return the length of the last sample in the current pack
        next_seq_len = (
            len(current_pack["tokens"][boundary:])
            if self.split_across_pack
            else current_pack["seq_lens"][-1]
        )

        return {
            "tokens": current_pack["tokens"][boundary:],
            "labels": current_pack["labels"][boundary:],
            "input_pos": current_pack["input_pos"][boundary:],
            "seq_lens": [next_seq_len],
        }

    def _add_pack(self, pack: PACK_TYPE) -> None:
        """Processes, pads and adds a pack to ``self.packs``."""
        pack = self._convert_to_tensors(pack)
        pack = self._pad_pack(pack, padding_idx=self.padding_idx)
        self.packs.append(pack)

    def _convert_to_tensors(self, pack: PACK_TYPE) -> PACK_TYPE:
        """Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors."""
        return {
            "tokens": torch.tensor(pack["tokens"], dtype=torch.long),
            "labels": torch.tensor(pack["labels"], dtype=torch.long),
            "input_pos": torch.tensor(pack["input_pos"], dtype=torch.long),
            "seq_lens": torch.tensor(pack["seq_lens"], dtype=torch.long),
        }

    def _pad_pack(self, pack: PACK_TYPE, padding_idx: int) -> PACK_TYPE:
        """Pads a pack to ``self.max_seq_len``."""
        # Pad tokens
        num_padding_tokens = self.max_seq_len - len(pack["tokens"])
        padded_tokens = F.pad(
            pack["tokens"],
            (0, num_padding_tokens),
            value=padding_idx,
        )

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(pack["tokens"])
        attention_mask = F.pad(
            attention_mask,
            (0, num_padding_tokens),
            value=0,
        )

        # Pad labels
        padded_labels = F.pad(
            pack["labels"],
            (0, self.max_seq_len - len(pack["labels"])),
            value=CROSS_ENTROPY_IGNORE_IDX,
        )

        # Add padding tokens as a last seq len to ensure sum is max_seq_len
        last_seq_len = pack["seq_lens"][-1] + num_padding_tokens
        padded_seq_lens = (
            torch.cat([pack["seq_lens"][:-1], torch.tensor([last_seq_len])])
            if num_padding_tokens > 0
            else pack["seq_lens"]
        )

        # Pad input_pos continuing the sequence from last value
        # in input_pos
        # e.g. [0 1 2] -> [0 1 2 3 4 5] for self.max_seq_len = 6
        num_range = torch.arange(
            pack["input_pos"][-1] + 1,
            pack["input_pos"][-1] + self.max_seq_len - len(pack["input_pos"]) + 1,
        )
        # Clamp to max_seq_len - 1 to avoid out of bounds error
        clamped_num_range = torch.clamp(num_range, 0, self.max_seq_len - 1)
        padded_input_pos = torch.cat([pack["input_pos"], clamped_num_range])

        return {
            "tokens": padded_tokens,
            "labels": padded_labels,
            "input_pos": padded_input_pos,
            "seq_lens": padded_seq_lens,
            "attention_mask": attention_mask,
        }

    def __len__(self) -> int:
        self._lazy_init()
        return len(self.packs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._lazy_init()
        return self.packs[idx]


class StatefulDistributedStreamingPackedDataset(IterableDataset):
    """
    A stateful distributed streaming dataset that packs multiple tokenized samples into fixed-length sequences on the fly.

    This is an iterable dataset designed for distributed training with state persistence capabilities.
    Unlike the standard PackedDataset which loads all data into memory, this dataset streams data
    on-demand while maintaining proper distributed sampling and state management for checkpointing.

    Args:
        ds (Dataset): dataset to sample pack. This should return a dictionary with field
            "tokens" and "labels" containing the tokenized and label samples.
        max_seq_len (int): Maximum number of tokens to pack
        padding_idx (int): padding index for the tokenizer. Default is 0.
        num_replicas (int): Number of processes participating in distributed data parallel (DDP) training.
            It should be the same as the world size of the DDP training.
        rank (int): Current process rank in DDP training.
        seed (int): Random seed for deterministic shuffling across epochs.
    """

    def __init__(
        self,
        ds: Dataset,
        *,
        max_seq_len: int,
        padding_idx: int = 0,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        self.ds = ds

        # For shuffling
        self.seed = seed
        self.epoch = 0

        # For packing with padding
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx

        # For resuming
        self.num_scanned_messages = 0

        # For distributed sampling and resuming
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_scanned_packs = 0
        # Used to handle the last pack to avoid deadlock in DDP training
        self.num_yielded_packs_for_rank = 0

        self.estimated_total_num_packs_for_rank = 0
        self._warm_up_to_estimate_total_num_packs_for_rank()

    def __len__(self) -> int:
        assert (
            self.estimated_total_num_packs_for_rank > 0
        ), "Please call _warm_up_to_estimate_total_num_packs_for_rank() first"
        return self.estimated_total_num_packs_for_rank

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
        then yield the buffer if it's the rank's turn. Continue until the end of the dataset."""

        # Adapted from torch.utils.data.distributed.DistributedSampler.__iter__
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.ds), generator=g).tolist()

        # Buffer to hold samples until they are long enough to be yielded
        current_pack = self._get_clean_pack()

        for idx in indices[self.num_scanned_messages :]:
            sample = self.ds[idx]
            tokens, labels = sample["tokens"], sample["labels"]

            # If the dataset outputs samples that are larger than the specified
            # max_seq_len and we're unable to split it, user needs to modify the parameter
            seq_len = len(tokens)
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"Dataset sample is too long ({seq_len} > {self.max_seq_len}). "
                    "Please increase `max_seq_len`."
                )

            if len(current_pack["tokens"]) + seq_len > self.max_seq_len:
                yield from self._pad_and_yield_pack_for_rank(current_pack)
                current_pack = self._get_clean_pack()

            current_pack["tokens"] += tokens
            current_pack["labels"] += labels
            current_pack["input_pos"] += list(range(seq_len))
            current_pack["seq_lens"] += [seq_len]
            self.num_scanned_messages += 1

        # Handle the last pack if there's leftover
        if len(current_pack["tokens"]) > 0:
            yield from self._pad_and_yield_pack_for_rank(current_pack)

        # Yield the last pack full of padding tokens if needed to avoid deadlock in DDP training
        if self.num_yielded_packs_for_rank * self.num_replicas < self.num_scanned_packs:
            yield from self._pad_and_yield_pack(
                {**self._get_clean_pack(), "seq_lens": [0]}
            )

        self.clear_all_states()

    def clear_all_states(self) -> None:
        self.num_scanned_messages = 0
        self.num_scanned_packs = 0
        self.num_yielded_packs_for_rank = 0

    # Copied from torch.utils.data.distributed.DistributedSampler.set_epoch
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def state_dict(self) -> dict[str, int]:
        return {
            "num_scanned_messages": self.num_scanned_messages,
            "num_scanned_packs": self.num_scanned_packs,
            "num_yielded_packs_for_rank": self.num_yielded_packs_for_rank,
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        def check_non_negative(key: str, state_dict: dict[str, int]) -> None:
            if key not in state_dict:
                raise ValueError(f"Invalid state_dict without {key}")
            if state_dict[key] < 0:
                raise ValueError(f"Cannot load state_dict with negative {key}")

        check_non_negative("num_scanned_messages", state_dict)
        self.num_scanned_messages = state_dict["num_scanned_messages"]

        check_non_negative("num_scanned_packs", state_dict)
        self.num_scanned_packs = state_dict["num_scanned_packs"]

        check_non_negative("num_yielded_packs_for_rank", state_dict)
        self.num_yielded_packs_for_rank = state_dict["num_yielded_packs_for_rank"]

    def _warm_up_to_estimate_total_num_packs_for_rank(
        self, num_packs_to_warm_up_for_rank: int = 3
    ) -> None:
        # Used to handle the situation where the total number of packs < num_packs_to_warm_up_for_rank
        num_packs_warmed_up_for_rank = 0
        # Unify the estimation across ranks
        original_rank = self.rank
        self.rank = self.num_replicas - 1

        def log_rank_zero(message: str):
            if 0 == original_rank:
                print(f"INFO({self.__class__.__name__}): {message}", flush=True)

        log_rank_zero("Start warm-up packing to estimate the total number of packs")

        for _ in iter(self):
            num_packs_warmed_up_for_rank += 1
            if num_packs_warmed_up_for_rank == num_packs_to_warm_up_for_rank:
                break
        self.estimated_total_num_packs_for_rank = int(
            len(self.ds) / self.num_scanned_messages * num_packs_warmed_up_for_rank
        )

        log_rank_zero(
            "Finish warm-up packing "
            f"(estimated_total_num_packs_for_rank = {self.estimated_total_num_packs_for_rank} "
            f"based on num_packs_warmed_up_for_rank = {num_packs_warmed_up_for_rank})"
        )

        # Restore all states to start fresh
        self.clear_all_states()
        self.rank = original_rank

    def _get_clean_pack(self) -> PACK_TYPE:
        return {
            "tokens": [],
            "labels": [],
            "input_pos": [],
            "seq_lens": [],
        }

    def _pad_and_yield_pack_for_rank(
        self, pack: PACK_TYPE
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Processes, pads and yields a pack if it's the rank's turn."""
        self.num_scanned_packs += 1
        if (self.num_scanned_packs - 1) % self.num_replicas == self.rank:
            self.num_yielded_packs_for_rank += 1
            yield from self._pad_and_yield_pack(pack)

    def _pad_and_yield_pack(self, pack: PACK_TYPE) -> Iterator[Dict[str, torch.Tensor]]:
        """Processes, pads and yields a pack."""
        pack = self._convert_to_tensors(pack)
        pack = self._pad_pack(pack, padding_idx=self.padding_idx)
        yield pack

    # Copied from PackedDataset._convert_to_tensors with more accurate type hints
    def _convert_to_tensors(self, pack: PACK_TYPE) -> dict[str, torch.Tensor]:
        """Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors."""
        return {
            "tokens": torch.tensor(pack["tokens"], dtype=torch.long),
            "labels": torch.tensor(pack["labels"], dtype=torch.long),
            "input_pos": torch.tensor(pack["input_pos"], dtype=torch.long),
            "seq_lens": torch.tensor(pack["seq_lens"], dtype=torch.long),
        }

    # Adapted from PackedDataset._pad_pack
    # with more accurate type hints and ability to handle empty input_pos
    def _pad_pack(
        self, pack: dict[str, torch.Tensor], padding_idx: int
    ) -> dict[str, torch.Tensor]:
        """Pads a pack to ``self.max_seq_len``."""
        # Pad tokens
        num_padding_tokens = self.max_seq_len - len(pack["tokens"])
        padded_tokens = F.pad(
            pack["tokens"],
            (0, num_padding_tokens),
            value=padding_idx,
        )

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(pack["tokens"])
        attention_mask = F.pad(
            attention_mask,
            (0, num_padding_tokens),
            value=0,
        )

        # Pad labels
        padded_labels = F.pad(
            pack["labels"],
            (0, self.max_seq_len - len(pack["labels"])),
            value=CROSS_ENTROPY_IGNORE_IDX,
        )

        # Add padding tokens as a last seq len to ensure sum is max_seq_len
        last_seq_len = pack["seq_lens"][-1] + num_padding_tokens
        padded_seq_lens = (
            torch.cat([pack["seq_lens"][:-1], torch.tensor([last_seq_len])])
            if num_padding_tokens > 0
            else pack["seq_lens"]
        )

        # Pad input_pos continuing the sequence from last value
        # in input_pos
        # e.g. [0 1 2] -> [0 1 2 3 4 5] for self.max_seq_len = 6
        num_range = torch.arange(
            pack["input_pos"][-1] + 1 if pack["input_pos"].numel() else 0,
            pack["input_pos"][-1] + self.max_seq_len - len(pack["input_pos"]) + 1
            if pack["input_pos"].numel()
            else self.max_seq_len,
        )
        # Clamp to max_seq_len - 1 to avoid out of bounds error
        clamped_num_range = torch.clamp(num_range, 0, self.max_seq_len - 1)
        padded_input_pos = torch.cat([pack["input_pos"], clamped_num_range])

        return {
            "tokens": padded_tokens,
            "labels": padded_labels,
            "input_pos": padded_input_pos,
            "seq_lens": padded_seq_lens,
            "attention_mask": attention_mask,
        }
