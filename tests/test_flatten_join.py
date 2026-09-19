"""A prompt built from several cached blocks flattens the way it did before blocks existed.

Regression for the 1.5 seam: a retrieval service and a downstream agent host hand
the LangChain adapter a
system prompt as a list of text blocks with cache markers. On the API runtimes the
blocks travel intact. On the agent runtimes they are flattened, and that flatten
joined the parts with one blank line in 0.1; joining with nothing glued
"instructions" and "corpus" into one word (found by that service's suite,
2026-09-13).
"""

from __future__ import annotations

from modelpass.adapters.anthropic import split_messages
from modelpass.types import CacheControl, Message, Role, TextBlock


def _system() -> Message:
    return Message(
        role=Role.SYSTEM,
        content=(
            TextBlock(text="instructions"),
            TextBlock(text="corpus", cache_control=CacheControl()),
        ),
    )


def test_text_is_byte_faithful_and_flat_text_keeps_the_blank_line():
    message = _system()
    assert message.text == "instructionscorpus"
    assert message.flat_text == "instructions\n\ncorpus"


def test_a_plain_string_reads_the_same_both_ways():
    message = Message(role=Role.USER, content="q")
    assert message.text == message.flat_text == "q"


def test_the_agent_runtime_flatten_uses_the_blank_line_join():
    system_prompt, prompt = split_messages([_system(), Message(role=Role.USER, content="q")])
    assert system_prompt == "instructions\n\ncorpus"
    assert prompt == "q"


def test_empty_blocks_do_not_leave_stray_separators():
    message = Message(role=Role.SYSTEM, content=(TextBlock(text=""), TextBlock(text="corpus")))
    assert message.flat_text == "corpus"
