# SPDX-License-Identifier: Apache-2.0
"""
MiniMax tool call parser for vllm-mlx.

Parses the MiniMax-M2 native XML tool call format:
<minimax:tool_call>
<invoke name="tool-name">
<parameter name="param-key">param-value</parameter>
</invoke>
</minimax:tool_call>
"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


def generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["minimax", "minimax_m2"])
class MiniMaxToolParser(ToolParser):
    """
    Parser for MiniMax-M2 tool call format.

    Format:
        <minimax:tool_call>
        <invoke name="func_name">
        <parameter name="key">value</parameter>
        </invoke>
        </minimax:tool_call>
    """

    TOOL_CALL_BLOCK = re.compile(
        r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
    )
    INVOKE_PATTERN = re.compile(
        r'<invoke\s+name="([^"]+)">(.*?)</invoke>', re.DOTALL
    )
    PARAM_PATTERN = re.compile(
        r'<parameter\s+name="([^"]+)">(.*?)</parameter>', re.DOTALL
    )
    THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        tool_calls: list[dict[str, Any]] = []

        # Find all tool call blocks
        blocks = self.TOOL_CALL_BLOCK.findall(model_output)
        if not blocks:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        for block in blocks:
            invokes = self.INVOKE_PATTERN.findall(block)
            for func_name, params_block in invokes:
                params = self.PARAM_PATTERN.findall(params_block)
                arguments = {}
                for p_name, p_value in params:
                    p_value = p_value.strip()
                    # Try to parse as JSON for proper typing
                    try:
                        arguments[p_name] = json.loads(p_value)
                    except (json.JSONDecodeError, ValueError):
                        arguments[p_name] = p_value

                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name.strip(),
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )

        # Extract content outside tool call blocks (strip thinking)
        cleaned = self.TOOL_CALL_BLOCK.sub("", model_output).strip()
        cleaned = self.THINK_PATTERN.sub("", cleaned).strip()
        # Remove trailing junk tokens like [e~[
        cleaned = re.sub(r"\[e~\[.*$", "", cleaned).strip()

        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=cleaned if cleaned else None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if "<minimax:tool_call>" not in current_text:
            return {"content": delta_text}

        if "</minimax:tool_call>" in delta_text:
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                return {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for i, tc in enumerate(result.tool_calls)
                    ]
                }

        return None
