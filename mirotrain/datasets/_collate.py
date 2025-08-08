# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX, PACK_TYPE
from torchtune.modules.attention_utils import packed_block_causal_mask


def padded_collate_packed(
    batch: List[PACK_TYPE],
    use_flash_attention: bool = False,
) -> Dict[str, torch.Tensor]:
    """Collate packed sequences into a batch. Only convert the seq lens into
    a block mask for use with attention. Tokens, labels, and input_pos are
    already padded to the same length within :class:`~torchtune.datasets.PackedDataset`.
    Args:
        batch (List[PACK_TYPE]): A list of pack dictionaries containing the following keys:
            - tokens: input token ids
            - labels: label token ids
            - input_pos: relative position ids for each sequence in pack
            - seq_lens: lengths of each sample within the pack
            - attention_mask: attention mask for each sample within the pack
        use_flash_attention (bool): Whether to use flash attention.
    Returns:
        Dict[str, torch.Tensor]: Collated input, label, input_pos, mask tensors.
    Example:
        >>> token_pairs = [
        >>>    {"tokens": [1, 2, 3, 4, 5, 6], "labels": [7, 8, 9, 10, 11, 12],
        >>>     "input_pos": [0, 1, 2, 0, 1, 0], "seq_lens": [3, 2, 1], "attention_mask": [1, 1, 1, 1, 1, 0]},
        >>>    {"tokens": [13, 14, 15, 16, 17, 18], "labels": [19, 20, 21, 22, 23, 24],
        >>>     "input_pos": [0, 1, 0, 1, 0, 1], "seq_lens": [2, 2, 2], "attention_mask": [1, 1, 1, 1, 0, 0]},
        >>> ]
        >>> collated = padded_collate_packed(
        >>>    batch=token_pairs,
        >>>    device=device,
        >>> )
        >>> collated["mask"]
        >>> tensor([
        >>> [[1, 0, 0, 0, 0, 0],
        >>>  [1, 1, 0, 0, 0, 0],
        >>>  [1, 1, 1, 0, 0, 0],
        >>>  [0, 0, 0, 1, 0, 0],
        >>>  [0, 0, 0, 1, 1, 0],
        >>>  [0, 0, 0, 0, 0, 1]],
        >>> [[1, 0, 0, 0, 0, 0],
        >>>  [1, 1, 0, 0, 0, 0],
        >>>  [0, 0, 1, 0, 0, 0],
        >>>  [0, 0, 1, 1, 0, 0],
        >>>  [0, 0, 0, 0, 1, 0],
        >>>  [0, 0, 0, 0, 1, 1]])
    """

    tokens = torch.stack([x["tokens"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    input_pos = torch.stack([x["input_pos"] for x in batch])

    if use_flash_attention:
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        seq_lens = [seq_len for x in batch for seq_len in x["seq_lens"]]

        # Generate max_seqlen tensor - record the longest sample length in each pack
        max_seqlen = max(seq_lens)
        # Generate cu_seqlens tensor - cumulative sequence lengths for packed data
        seq_lens = [0] + seq_lens
        cu_seqlens = torch.cumsum(torch.tensor(seq_lens, dtype=torch.int32), dim=0).to(
            torch.int32
        )

        return {
            "tokens": tokens,
            "labels": labels,
            "input_pos": input_pos,
            "max_seqlen": max_seqlen,
            "cu_seqlens": cu_seqlens,
            "attention_mask": attention_mask,
        }
    else:
        seq_lens = [x["seq_lens"] for x in batch]

        block_mask = packed_block_causal_mask(
            seq_lens=seq_lens,
        )

        return {
            "tokens": tokens,
            "labels": labels,
            "input_pos": input_pos,
            "mask": block_mask,
        }


def padded_collate_dpo(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    pad_to_multiple_of: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of sequences for Direct Preference Optimization (DPO).

    This function takes a batch of sequences, where each sequence is represented
    as a dictionary with multiple key-value pairs. Each key corresponds to a different
    sequence component, such as input_ids or labels.

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries, where each dictionary
            represents a sequence with multiple components, 'chosen_input_ids',
            'chosen_labels', 'rejected_input_ids', and 'rejected_labels' are required.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.
        pad_to_multiple_of (int): If > 1, pad the sequence to a multiple of this number.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing concatenated and padded
        input ids, labels, and attention mask.

    Example:
        >>> batch = [
        >>>    {'chosen_input_ids': [1, 2, 3], 'rejected_input_ids': [4, 5],
        >>>      'chosen_labels': [6, 7, 8], 'rejected_labels': [9, 10]},
        >>>    {'chosen_input_ids': [11, 12], 'rejected_input_ids': [13, 14, 15],
        >>>      'chosen_labels': [16, 17], 'rejected_labels': [18, 19, 20]},
        >>> ]
        >>> padded_collate_dpo(batch)
        >>> (tensor([[ 1,  2,  3],
        >>>          [11, 12,  0],
        >>>          [ 4,  5,  0],
        >>>          [13, 14, 15]]),
        >>>  tensor([[ 6,  7,  8],
        >>>          [16, 17, -100],
        >>>          [ 9, 10, -100],
        >>>          [18, 19, 20]]),
        >>>  tensor([[ 1,  1,  1],
        >>>          [ 1,  1,  0],
        >>>          [ 1,  1,  0],
        >>>          [ 1,  1,  1]]))
    """
    chosen_input_ids = [torch.tensor(ex["chosen_input_ids"]) for ex in batch]
    rejected_input_ids = [torch.tensor(ex["rejected_input_ids"]) for ex in batch]
    chosen_labels = [torch.tensor(ex["chosen_labels"]) for ex in batch]
    rejected_labels = [torch.tensor(ex["rejected_labels"]) for ex in batch]

    to_pad_input_ids = chosen_input_ids + rejected_input_ids
    to_pad_labels = chosen_labels + rejected_labels

    concatenated_input_ids = pad_sequence(
        to_pad_input_ids, batch_first=True, padding_value=padding_idx
    )
    concatenated_labels = pad_sequence(
        to_pad_labels, batch_first=True, padding_value=ignore_idx
    )

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (concatenated_input_ids != padding_idx).long()

    # Pad to multiple of N
    if pad_to_multiple_of > 1:
        concatenated_input_ids = F.pad(
            concatenated_input_ids,
            (
                0,
                pad_to_multiple_of
                - (concatenated_input_ids.size(1) % pad_to_multiple_of),
            ),
            value=padding_idx,
        )
        concatenated_labels = F.pad(
            concatenated_labels,
            (
                0,
                pad_to_multiple_of - (concatenated_labels.size(1) % pad_to_multiple_of),
            ),
            value=ignore_idx,
        )
        attention_mask = F.pad(
            attention_mask,
            (
                0,
                pad_to_multiple_of - (attention_mask.size(1) % pad_to_multiple_of),
            ),
            value=0,
        )

    return concatenated_input_ids, concatenated_labels, attention_mask
