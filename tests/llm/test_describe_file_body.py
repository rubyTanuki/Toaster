from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from typing import Type

from pydantic import BaseModel

from tostr.semantic.llm.base import LLMClient, LLMStrategy
from tostr.semantic.describer import LLMDescriber
from tostr.core.models import BaseFile
from tostr.graph.registry import Registry


class _BodyStrategy(LLMStrategy):
    """Records the input it was given and returns a fixed description."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_input: str | None = None
        self.last_system: str | None = None

    async def generate(self, input_data_string: str, system_instruction: str, response_schema: Type[BaseModel]) -> BaseModel:
        self.last_input = input_data_string
        self.last_system = system_instruction
        return response_schema(description="A login page with a sign-in form.")


def _make_file(uid: str, body: str) -> BaseFile:
    registry = MagicMock(spec=Registry)
    registry.progress_tracker = None
    f = BaseFile(name=uid, uid=uid)
    f.registry = registry
    f.body = body
    f.diff_hash = "abc123"
    return f


@pytest.mark.asyncio
async def test_childless_file_with_body_is_described_from_body():
    strategy = _BodyStrategy(api_key="t", model_name="t")
    embedder = MagicMock()
    describer = LLMDescriber(llm=LLMClient(strategy), embedder=embedder)

    html = "<html><body><form><input name='user'></form></body></html>"
    f = _make_file("index.html", html)

    await describer.describe(f)

    # Description came from the body path, not the "Empty File" tag.
    assert f.description == "A login page with a sign-in form."
    assert "Empty File" not in (f.description or "")
    # The raw body was actually sent to the LLM (this is the whole point).
    assert "<form>" in strategy.last_input
    # The file was enqueued for embedding so search has signal.
    embedder.enqueue.assert_called_once_with(f)


@pytest.mark.asyncio
async def test_truly_empty_file_falls_back_to_empty_tag():
    strategy = _BodyStrategy(api_key="t", model_name="t")
    embedder = MagicMock()
    describer = LLMDescriber(llm=LLMClient(strategy), embedder=embedder)

    f = _make_file("blank.html", "   \n  ")
    await describer.describe(f)

    assert f.description == "Empty File"
    # No LLM call for an empty file.
    assert strategy.last_input is None
