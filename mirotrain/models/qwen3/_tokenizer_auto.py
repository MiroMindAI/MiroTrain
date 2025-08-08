# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Mapping, Optional, Tuple

from torchtune.data import truncate
from torchtune.data._messages import Message
from transformers import AutoTokenizer


class Qwen3TokenizerAuto:  # Tokenizer wrapper for Qwen3
    """This tokenizer is simply a wrapper around the Qwen2_5Tokenizer, with a few extra tokens added.

    See <https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/qwen2/tokenization_qwen2.py>
    and <https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json>.

    Args:
        path (str): Path to the Qwen3 model for loading tokenizer.
        max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
            Default: None
        truncation_type (str): type of truncation to apply, either "left" or "right".
            Default is "right".

    Example:
        >>> tokenizer = Qwen3TokenizerAuto(
                path="/path/to/Qwen3")
        >>> tokenized_text = tokenizer.encode("Hello world!")
        >>> print(tokenized_text)
        [39, 385, 78, 675, 0, 2000]
    """

    def __init__(
        self,
        path: str,
        max_seq_len: Optional[int] = None,
        truncation_type: str = "right",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.max_seq_len = max_seq_len
        self.truncation_type = truncation_type

        # ID for <|im_end|> token, used for optional truncation and alignment
        self.eos_id = self.tokenizer.eos_token_id

        # ID for "<|endoftext|> token, used for pad
        self.pad_id = self.tokenizer.pad_token_id

        # Tokens that prefix assistant messages: <|im_start|>assistant\n
        self.assistant_prefix_tokens = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self.assistant_think_tags = "<think>\n\n</think>\n\n"
        self.assistant_think_tokens = self.tokenizer.encode(
            self.assistant_think_tags, add_special_tokens=False
        )

        self.decode = self.tokenizer.decode
        self.encode = self.tokenizer.encode

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Tokenize the 'messages' field in a sample, and add 'tokens' and 'mask' fields.

        Args:
            sample (Mapping[str, Any]): Must contain a "messages" field.

        Returns:
            Mapping[str, Any]: Modified sample with "tokens" and "mask".

        Raises:
            KeyError: If the 'messages' field is missing from the sample.
        """
        if "messages" not in sample:
            raise KeyError("Missing 'messages' field in sample.")

        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample

    def tokenize_messages(
        self,
        messages: List[Message],
        add_eos: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Tokenize a list of Message objects into input IDs and label mask.

        Args:
            messages (List[Message]): Structured conversation messages.
            add_eos (bool): Whether to append <|im_end|> at the end for truncation.

        Returns:
            Tuple[List[int], List[bool]]:
                - input_ids: Tokenized conversation tokens.
                - label_mask: True where token is ignored in loss.

        Raises:
            ValueError: If a message contains unsupported content types
                        or the role is 'ipython'.
        """
        input_ids: List[int] = []
        label_mask: List[bool] = []

        for msg in messages:
            if msg.role == "ipython":
                raise ValueError("ipython role not supported in tokenize_messages.")

            # Combine content text pieces (only "text" type supported)
            text_parts = []
            for item in msg.content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    raise ValueError(f"Unsupported message content type: {item}")
                text_parts.append(item.get("content", ""))
            msg_text = "".join(text_parts)

            # Manually build chat template
            if msg.role == "assistant":
                # Check if msg_text already starts with <think>
                has_think_content = msg_text.strip().startswith("<think>")

                if has_think_content:
                    # If msg_text already has <think> content, don't add extra think tags
                    full_text = "<|im_start|>assistant\n" + msg_text + "<|im_end|>\n"
                else:
                    # Original logic: add assistant_think_tags
                    full_text = (
                        "<|im_start|>assistant\n"
                        + self.assistant_think_tags
                        + msg_text
                        + "<|im_end|>\n"
                    )
            else:
                full_text = f"<|im_start|>{msg.role}\n{msg_text}<|im_end|>\n"

            tokens = self.tokenizer.encode(full_text, add_special_tokens=False)

            # Append tokens
            start = len(input_ids)
            input_ids.extend(tokens)
            label_mask.extend([False] * len(tokens))  # Default: not masked

            # Mask entire message if marked
            if msg.masked:
                for i in range(start, start + len(tokens)):
                    label_mask[i] = True

            # Mask assistant prefix and <think> block
            if msg.role == "assistant":
                # Mask <|im_start|>assistant\n
                for i in range(start, start + len(self.assistant_prefix_tokens)):
                    label_mask[i] = True

                if has_think_content:
                    # If msg_text already has <think> content, preserve the think tags and content
                    # Since <think> comes right after <|im_start|>assistant\n, we don't need to mask anything else
                    # The <think> tags and content will be preserved (not masked)
                    pass
                else:
                    # Original logic: mask the added <think>\n\n</think>\n\n
                    for i in range(
                        start + len(self.assistant_prefix_tokens),
                        start
                        + len(self.assistant_prefix_tokens)
                        + len(self.assistant_think_tokens),
                    ):
                        label_mask[i] = True

        # Optional: remove trailing \n if <|im_end|> is not last
        if input_ids and input_ids[-1] != self.eos_id:
            input_ids = input_ids[:-1]
            label_mask = label_mask[:-1]

        # Truncate input and mask if needed
        if self.max_seq_len is not None:
            input_ids = truncate(
                tokens=input_ids,
                max_seq_len=self.max_seq_len,
                eos_id=self.eos_id if add_eos else None,
                truncation_type=self.truncation_type,
            )
            label_mask = truncate(
                tokens=label_mask,
                max_seq_len=self.max_seq_len,
                eos_id=True if add_eos else None,  # Use True as placeholder
                truncation_type=self.truncation_type,
            )

        return input_ids, label_mask
