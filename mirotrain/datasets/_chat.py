# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional, Union

from mirotrain.data import TracesToMessages

from torchtune.data import OpenAIToMessages, ShareGPTToMessages
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer

from ._packed import PackedDataset


def odr_chat_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    conversation_column: str,
    conversation_style: str,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    masking_strategy: Optional[str] = "train_on_assistant",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    Configure a custom dataset with conversations between user and model assistant.

    This builder function can be used to configure a custom chat dataset directly from the yaml config
    as an alternative to :class:`~torchtune.datasets.SFTDataset`, as it is made to be config friendly.

    The dataset is expected to contain a single column with the conversations:

    .. code-block:: text

        |  conversations                         |
        |----------------------------------------|
        | [{"role": "user", "content": Q1},      |
        |  {"role": "assistant", "content": A1}] |

    This will be converted to:

    .. code-block:: python

        messages = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]

    This list of messages is then tokenized for model training.

    You may have a different structure for your conversations, such as different role names or
    different keys in the json structure. You can use the ``conversation_style`` parameter
    to choose from standard formats such as "sharegpt" (see :class:`~torchtune.data.ShareGPTToMessages`)
    or "openai" (see :class:`~torchtune.data.OpenAIToMessages`). If your dataset is not in one of these
    formats, we recommend creating a custom message transform and using it in a custom dataset
    builder function similar to :class:`~torchtune.datasets.chat_dataset`.

    If your column names are different, use the ``conversation_column`` parameter to point
    towards the column with the conversations.

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is
    set to ``False`` by default.

    - If ``train_on_input`` is True, the prompt is used during training and
      contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100).

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text"), pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        conversation_column (str): name of column containing the conversations.
        conversation_style (str): string specifying expected style of conversations in the dataset
            for automatic conversion to the :class:`~torchtune.data.Message` structure.
            Supported styles are: "sharegpt", "openai", "traces"
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Default is None.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        masking_strategy (Optional[str]): masking strategy to use for model training.
            Must be one of: `train_on_all`, `train_on_assistant`, `train_on_last`.
            Default is "train_on_assistant".

            - ``train_on_all``: both user and assistant messages are unmasked
            - ``train_on_assistant``: user messages are masked, only assistant messages are unmasked
            - ``train_on_last``: only the last assistant message is unmasked
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``,
            such as ``data_files`` or ``split``.

    Returns:
        Union[SFTDataset, PackedDataset]: the configured :class:`~torchtune.datasets.SFTDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``

    Raises:
        ValueError: if the conversation format is not supported
    """
    if conversation_style == "sharegpt":
        message_transform = ShareGPTToMessages(
            train_on_input=train_on_input,
            column_map={"conversations": conversation_column},
            new_system_prompt=new_system_prompt,
            masking_strategy=masking_strategy,
        )
    elif conversation_style == "openai":
        message_transform = OpenAIToMessages(
            train_on_input=train_on_input,
            column_map={"messages": conversation_column},
            new_system_prompt=new_system_prompt,
            masking_strategy=masking_strategy,
        )
    elif conversation_style == "traces":
        message_transform = TracesToMessages(
            train_on_input=train_on_input,
            column_map={"conversations": conversation_column},
            new_system_prompt=new_system_prompt,
            masking_strategy=masking_strategy,
        )
    else:
        raise ValueError(f"Unsupported conversation style: {conversation_style}")

    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        split=split,
        filter_fn=filter_fn,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds
