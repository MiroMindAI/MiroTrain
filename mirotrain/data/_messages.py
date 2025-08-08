# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Mapping, Optional
from warnings import warn

from torchtune.data._messages import mask_messages, Message, Transform
from torchtune.data._utils import format_content_with_images, load_image


__targets__ = ("torchtune.data._messages", "torchtune.datasets._sft")
__implements__ = ("validate_messages",)


class TracesToMessages(Transform):
    """
    Convert a single chat sample adhering to thes Traces JSON structure to torchtune's :class:`~torchtune.data.Message`
    structure.

    A single sample typically consists of a single optional system prompt and one or multiple
    turns of user and assistant messages.

    Traces follows::

        {
            "messages": [
                {
                    "role": <system|user|assistant>,
                    "content": <message>,
                },
                ...
            ]
        }

    :class:`~torchtune.data.Message` follows::

        [
            {
                "role": <system|user|assistant>,
                "content": <message>,
            },
            ...
        ]

    Args:
        train_on_input (Optional[bool]): whether the model is trained on the user prompt or not.
            Deprecated parameter and will be removed in a future release.
            Default is None.
        column_map (Optional[dict[str, str]]): a mapping from the expected columns ("conversations")
            to the new column names in the dataset. Key should be "conversations" and value should
            be the new column name. If None, keep the default "conversations".
            Default is None.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.
        image_dir (Optional[Path]): path to the directory containing the images that is prepended to all image
            paths in the dataset. For example, if ``image_dir="/home/user/dataset/"` and the sample image path
            was ``"images/1.jpg"``, the final image path that will be loaded is ``"/home/user/dataset/images/1.jpg"``.
            If None, assume images are available in current working directory or are located
            on a remote url. For text-only, leave as None. Default is None.
        image_tag (Optional[str]): placeholder tags in the text content of each message to be replaced by image
            special tokens. If images are present and this is None, then will prepend image tokens to the first
            user message in the sample by default. If text-only, this field is ignored. Default is ``"<image>"``.
        masking_strategy (Optional[str]): masking strategy to use for model training.
            Must be one of: `train_on_all`, `train_on_assistant`, `train_on_last`.
            Default is "train_on_assistant".

            - ``train_on_all``: both user and assistant messages are unmasked
            - ``train_on_assistant``: user messages are masked, only assistant messages are unmasked
            - ``train_on_last``: only the last assistant message is unmasked

            Note: Multimodal user messages are always masked.

    Raises:
        ValueError: If ``column_map`` is provided and ``conversations`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: Optional[bool] = None,
        column_map: Optional[dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
        image_dir: Optional[Path] = None,
        image_tag: Optional[str] = "<image>",
        masking_strategy: Optional[str] = "train_on_assistant",
    ):
        if train_on_input is True:
            warn(
                "train_on_input is deprecated and will be removed in a future release. "
                "Please use masking_strategy instead."
                "You should replace train_on_input=True with masking_strategy='train_on_all', and "
                "train_on_input=False with masking_strategy='train_on_assistant'."
                "For backwards compatibility, if you pass both train_on_input and masking_strategy, "
                "the value of masking_strategy will be ignored until torchtune 0.7. ",
                DeprecationWarning,
                stacklevel=2,
            )

            masking_strategy = (
                "train_on_all" if train_on_input else "train_on_assistant"
            )
        self.masking_strategy = masking_strategy
        self.new_system_prompt = new_system_prompt
        if column_map:
            if "conversations" not in column_map:
                raise ValueError(
                    f"Expected a key of 'conversations' in column_map but found {column_map.keys()}."
                )
            self._column_map = column_map
        else:
            self._column_map = {"conversations": "conversations", "image": "image"}
        self.image_dir = image_dir
        self.image_tag = image_tag

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Return a list of Message objects from the provided sample dict.

        Args:
            sample (Mapping[str, Any]): a single data sample with "messages" field pointing
                to a list of dict messages.

        Returns:
            list[Message]: A list of messages with "role" and "content" fields.
        """
        role_map = {"system": "system", "user": "user", "assistant": "assistant"}
        messages = []
        if self.new_system_prompt is not None:
            messages.append(
                Message(
                    role="system", content=self.new_system_prompt, masked=True, eot=True
                )
            )

        is_multimodal = "image" in sample or (
            "image" in self._column_map and self._column_map["image"] in sample
        )

        # Gate variable to ensure that we only prepend image tokens to the first user message
        image_loaded = False
        for message in sample[self._column_map["conversations"]]:
            role = role_map[message["role"]]
            content = message["content"]
            if role == "system" and self.new_system_prompt is not None:
                continue
            if role == "user":
                if is_multimodal and not image_loaded:
                    image_path = sample[self._column_map["image"]]
                    if self.image_dir is not None:
                        image_path = self.image_dir / image_path
                    pil_image = load_image(image_path)
                    # If image tag is not specified, prepend by default
                    if self.image_tag is None:
                        content = [
                            {"type": "image", "content": pil_image},
                            {"type": "text", "content": content},
                        ]
                    else:
                        content = format_content_with_images(
                            content,
                            image_tag=self.image_tag,
                            images=[pil_image],
                        )
                    image_loaded = True
            messages.append(Message(role=role, content=content))
        mask_messages(messages, self.masking_strategy)

        return {"messages": messages}


def validate_messages(messages: list[Message]) -> None:
    """
    Given a list of messages, ensure that messages form a valid
    back-and-forth conversation. An error will be raised if:

    - There is a system message that's not the first message
    - [DISABLED] There are two consecutive user messages
    - An assistant message comes before the first user message
    - The message is empty
    - Messages are shorter than length of 2 (min. one user-assistant turn)


    Args:
        messages (list[Message]): the messages to validate.

    Raises:
        ValueError: If the messages are invalid.
    """
    if len(messages) < 2:
        raise ValueError(
            f"Messages must be at least length 2, but got {len(messages)} messages"
        )

    last_message = Message(role="assistant", content="")
    for i, message in enumerate(messages):
        if message.role == "assistant" and last_message.role not in [
            "user",
            "tool",
            "ipython",
        ]:
            raise ValueError(
                f"Assistant message before expected user, tool or ipython message at index {i} in messages"
            )
        # if message.role == "user" and last_message.role == "user":
        #     raise ValueError(
        #         f"Two consecutive user messages at index {i} and {i - 1} in messages"
        #     )
        if message.role == "system" and i > 0:
            raise ValueError(
                f"System message at index {i} in messages, but system messages must come first"
            )
        if message.role in ["tool", "ipython"] and not last_message.ipython:
            raise ValueError(
                f"Tool or ipython message at index {i} must follow an ipython message"
            )
        last_message = message
