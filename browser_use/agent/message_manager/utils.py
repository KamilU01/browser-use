from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, Type

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)


def extract_json_from_model_output(content: str) -> dict:
    """Extract JSON from model output, handling both plain JSON and code-block-wrapped JSON."""
    try:
        # If content is wrapped in code blocks, extract just the JSON part
        if "```" in content:
            # Find the JSON content between code blocks
            content = content.split("```")[1]
            # Remove language identifier if present (e.g., 'json\n')
            if "\n" in content:
                content = content.split("\n", 1)[1]
        # Parse the cleaned content
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse model output: {content} {str(e)}")
        raise ValueError("Could not parse response.")


def convert_input_messages(
    input_messages: list[BaseMessage], model_name: Optional[str]
) -> list[BaseMessage]:
    """Convert input messages to a format that is compatible with the planner model"""
    if model_name is None:
        return input_messages
    if (
        model_name == "deepseek-reasoner"
        or "deepseek-r1" in model_name
        or "gemma" in model_name
    ):
        converted_input_messages = _convert_messages_for_non_function_calling_models(
            input_messages
        )
        merged_input_messages = _merge_successive_messages(
            converted_input_messages, HumanMessage
        )
        merged_input_messages = _merge_successive_messages(
            merged_input_messages, AIMessage
        )
        return merged_input_messages
    """ if 'gemma' in model_name.lower():
		return _convert_messages_for_gemma(input_messages) """
    return input_messages


def _convert_messages_for_non_function_calling_models(
    input_messages: list[BaseMessage],
) -> list[BaseMessage]:
    """Convert messages for non-function-calling models"""
    output_messages = []
    for message in input_messages:
        if isinstance(message, HumanMessage):
            output_messages.append(message)
        elif isinstance(message, SystemMessage):
            output_messages.append(message)
        elif isinstance(message, ToolMessage):
            output_messages.append(HumanMessage(content=message.content))
        elif isinstance(message, AIMessage):
            # check if tool_calls is a valid JSON object
            if message.tool_calls:
                tool_calls = json.dumps(message.tool_calls)
                output_messages.append(AIMessage(content=tool_calls))
            else:
                output_messages.append(message)
        else:
            raise ValueError(f"Unknown message type: {type(message)}")
    return output_messages


def _convert_messages_for_gemma(input_messages: list[BaseMessage]) -> list[BaseMessage]:
    """Convert messages for Gemma models to ensure proper user/assistant alternation"""
    # Start with a single system message if present
    converted_messages = []
    has_system = False

    # Handle system message (if present)
    if input_messages and isinstance(input_messages[0], SystemMessage):
        system_content = input_messages[0].content
        converted_messages.append(
            HumanMessage(content=f"System instructions: {system_content}")
        )
        has_system = True
        start_idx = 1
    else:
        start_idx = 0

    # Process remaining messages ensuring alternation
    last_role_is_human = True  # We start with a human message

    for i in range(start_idx, len(input_messages)):
        msg = input_messages[i]
        is_human_message = isinstance(msg, HumanMessage) or isinstance(msg, ToolMessage)

        # If we need to maintain alternation and current message would break it
        if is_human_message == last_role_is_human:
            # If the next message should be AI but we have a human message
            if last_role_is_human:
                # Insert an empty AI message to maintain alternation
                converted_messages.append(AIMessage(content="I'll process that."))
            else:
                # Insert an empty human message to maintain alternation
                converted_messages.append(HumanMessage(content="Please continue."))

        # Add the current message with the appropriate type
        if is_human_message:
            if isinstance(msg, ToolMessage):
                converted_messages.append(
                    HumanMessage(content=f"Tool result: {msg.content}")
                )
            else:
                converted_messages.append(msg)
            last_role_is_human = True
        else:
            converted_messages.append(msg)
            last_role_is_human = False

    # Ensure we end with an even number of messages (for next AI response)
    if len(converted_messages) % 2 == 0:
        converted_messages.append(
            HumanMessage(content="Please provide your next response.")
        )

    return converted_messages


def _merge_successive_messages(
    messages: list[BaseMessage], class_to_merge: Type[BaseMessage]
) -> list[BaseMessage]:
    """Some models like deepseek-reasoner dont allow multiple human messages in a row. This function merges them into one."""
    merged_messages = []
    streak = 0
    for message in messages:
        if isinstance(message, class_to_merge):
            streak += 1
            if streak > 1:
                if isinstance(message.content, list):
                    merged_messages[-1].content += message.content[0][
                        "text"
                    ]  # type:ignore
                else:
                    merged_messages[-1].content += message.content
            else:
                merged_messages.append(message)
        else:
            merged_messages.append(message)
            streak = 0
    return merged_messages


def save_conversation(
    input_messages: list[BaseMessage],
    response: Any,
    target: str,
    encoding: Optional[str] = None,
) -> None:
    """Save conversation history to file."""

    # create folders if not exists
    if dirname := os.path.dirname(target):
        os.makedirs(dirname, exist_ok=True)

    with open(
        target,
        "w",
        encoding=encoding,
    ) as f:
        _write_messages_to_file(f, input_messages)
        _write_response_to_file(f, response)


def _write_messages_to_file(f: Any, messages: list[BaseMessage]) -> None:
    """Write messages to conversation file"""
    for message in messages:
        f.write(f" {message.__class__.__name__} \n")

        if isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    f.write(item["text"].strip() + "\n")
        elif isinstance(message.content, str):
            try:
                content = json.loads(message.content)
                f.write(json.dumps(content, indent=2) + "\n")
            except json.JSONDecodeError:
                f.write(message.content.strip() + "\n")

        f.write("\n")


def _write_response_to_file(f: Any, response: Any) -> None:
    """Write model response to conversation file"""
    f.write(" RESPONSE\n")
    f.write(
        json.dumps(json.loads(response.model_dump_json(exclude_unset=True)), indent=2)
    )
