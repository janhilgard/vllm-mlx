# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for GPT-OSS models using channel-based format.

GPT-OSS models use a channel-based token format instead of <think>...</think> tags:
    <|channel|>analysis<|message|>[reasoning]<|start|>assistant<|channel|>final<|message|>[content]<|return|>

This parser extracts reasoning from the 'analysis' channel and content from
the 'final' channel, stripping all structural tokens from API responses.
"""

import re

from .base import DeltaMessage, ReasoningParser

# Structural tokens used by GPT-OSS models
_STRUCTURAL_TOKENS = re.compile(
    r"<\|start\|>|<\|end\|>|<\|channel\|>|<\|return\|>|<\|call\|>"
)

# Markers for channel boundaries
_ANALYSIS_MARKER = "<|channel|>analysis<|message|>"
_FINAL_MARKER = "<|channel|>final<|message|>"


def _extract_channel(text: str, channel_name: str) -> str | None:
    """
    Extract content from a named channel.

    Finds <|channel|>{name}<|message|> and extracts text up to the next
    structural token or end of string.

    Args:
        text: Full model output text.
        channel_name: Channel name to extract (e.g., "analysis", "final").

    Returns:
        Extracted channel content, or None if channel not found.
    """
    marker = f"<|channel|>{channel_name}<|message|>"
    idx = text.find(marker)
    if idx == -1:
        return None

    start = idx + len(marker)
    rest = text[start:]

    # Find next structural token
    match = _STRUCTURAL_TOKENS.search(rest)
    content = rest[: match.start()] if match else rest

    return content if content else None


class GptOssReasoningParser(ReasoningParser):
    """
    Reasoning parser for GPT-OSS models.

    GPT-OSS uses channel-based tokens:
        <|channel|>analysis<|message|>[reasoning]
        <|start|>assistant<|channel|>final<|message|>[content]<|return|>

    The 'analysis' channel maps to reasoning, 'final' to content.

    Example:
        Input:  "<|channel|>analysis<|message|>Let me think<|start|>assistant
                 <|channel|>final<|message|>The answer is 42<|return|>"
        Output: reasoning="Let me think", content="The answer is 42"
    """

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning and content from complete model output.

        Args:
            model_output: Complete text output from the model.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        # No channel tokens at all — pure content fallback
        if "<|channel|>" not in model_output:
            return None, model_output

        reasoning = _extract_channel(model_output, "analysis")
        content = _extract_channel(model_output, "final")

        # Strip trailing <|return|> from content if present
        if content and content.endswith("<|return|>"):
            content = content[: -len("<|return|>")]

        # If we found channel tokens but no final channel, strip all
        # structural tokens and return as content
        if content is None and reasoning is None:
            cleaned = _STRUCTURAL_TOKENS.sub("", model_output)
            cleaned = cleaned.replace("<|channel|>", "").replace("<|message|>", "")
            return None, cleaned.strip() or None

        return (
            reasoning.strip() if reasoning else None,
            content.strip() if content else None,
        )

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta.

        Uses stateless phase detection from current_text on each call.

        Args:
            previous_text: Accumulated text before this delta.
            current_text: Accumulated text including this delta.
            delta_text: Just the new text in this streaming chunk.

        Returns:
            DeltaMessage with reasoning/content, or None to suppress.
        """
        prev_phase = self._detect_phase(previous_text)
        curr_phase = self._detect_phase(current_text)

        # Phase: init — no channel markers yet, suppress output
        if curr_phase == "init":
            return None

        # Phase: transition — between channels, suppress structural tokens
        if curr_phase == "transition":
            return None

        # Phase changed — extract only the relevant part from delta
        if prev_phase != curr_phase:
            content_after = self._extract_content_after_marker_in_delta(
                current_text, curr_phase
            )
            if not content_after:
                return None
            if curr_phase == "analysis":
                return DeltaMessage(reasoning=content_after)
            elif curr_phase == "final":
                return DeltaMessage(content=self._strip_return(content_after))
            return None

        # Same phase as before — pass delta through
        if curr_phase == "analysis":
            return DeltaMessage(reasoning=delta_text)
        elif curr_phase == "final":
            cleaned = self._strip_return(delta_text)
            return DeltaMessage(content=cleaned) if cleaned else None

        return None

    @staticmethod
    def _detect_phase(text: str) -> str:
        """
        Detect current streaming phase from accumulated text.

        Returns:
            "final"      — final channel marker complete
            "analysis"   — analysis marker complete, no structural token after
            "transition" — analysis present but structural token follows
            "init"       — no channel marker yet
        """
        final_idx = text.find(_FINAL_MARKER)
        if final_idx != -1:
            return "final"

        analysis_idx = text.find(_ANALYSIS_MARKER)
        if analysis_idx != -1:
            after_analysis = text[analysis_idx + len(_ANALYSIS_MARKER) :]
            if _STRUCTURAL_TOKENS.search(after_analysis):
                return "transition"
            return "analysis"

        return "init"

    @staticmethod
    def _extract_content_after_marker_in_delta(
        current_text: str, phase: str
    ) -> str | None:
        """
        When phase changes, extract only the content after the phase marker
        that falls within the current accumulated text's tail.

        Args:
            current_text: Full accumulated text.
            phase: Current phase ("analysis" or "final").

        Returns:
            Content after the marker, or None.
        """
        if phase == "analysis":
            marker = _ANALYSIS_MARKER
        elif phase == "final":
            marker = _FINAL_MARKER
        else:
            return None

        idx = current_text.find(marker)
        if idx == -1:
            return None

        after = current_text[idx + len(marker) :]

        # For analysis, stop at next structural token
        if phase == "analysis":
            match = _STRUCTURAL_TOKENS.search(after)
            if match:
                after = after[: match.start()]

        return after if after else None

    @staticmethod
    def _strip_return(text: str) -> str:
        """Strip <|return|> from text."""
        return text.replace("<|return|>", "")
