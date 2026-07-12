from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from tostr.semantic.llm.base import LLMClient, LLMStrategy, LLMResponse
from tostr.semantic.describer import LLMDescriber
from tostr.core.models import BaseClass, BaseMethod, BaseFile
from tostr.graph.registry import Registry
from pydantic import BaseModel, Field
from typing import Type

class MockStrategy(LLMStrategy):
    async def generate(self, input_data_string: str, system_instruction: str, response_schema: Type[BaseModel]) -> BaseModel:
        # We need to return an instance of response_schema
        data = {}
        if "description" in response_schema.model_fields:
            data["description"] = "Mock class description"
        if "description_map" in response_schema.model_fields:
            data["description_map"] = {"0": "Mock method description"}
        return response_schema(**data)

@pytest.mark.asyncio
async def test_llm_client_generate_description():
    """
    Test that LLMClient correctly calls the strategy and returns the parsed response.
    """
    strategy = MockStrategy(api_key="test", model_name="test")
    client = LLMClient(strategy)
    
    class TestSchema(BaseModel):
        description: str
        
    response = await client.generate_description({"input": "data"}, "instruction", TestSchema)
    
    assert isinstance(response, TestSchema)
    assert response.description == "Mock class description"

@pytest.mark.asyncio
async def test_models_resolve_description_data_flow():
    """
    Test the full data flow from LLMDescriber through LLMClient into BaseClass/BaseMethod.
    """
    registry = MagicMock(spec=Registry)
    registry.progress_tracker = None
    parent_file = MagicMock(spec=BaseFile, imports=[], package="com.example")

    cls = BaseClass(name="TestClass", uid="TestClass")
    cls.registry = registry
    cls.parent = parent_file

    method = BaseMethod(name="testMethod", uid="TestClass#testMethod")
    method.registry = registry
    cls.add_child(method)

    cls.skeletonize = MagicMock(return_value="class skeleton")

    strategy = MockStrategy(api_key="test", model_name="test")
    client = LLMClient(strategy)
    embedder = MagicMock()

    describer = LLMDescriber(llm=client, embedder=embedder)
    await describer.describe(cls)

    assert cls.description == "Mock class description"
    assert method.description == "Mock method description"

@pytest.mark.asyncio
async def test_llm_client_retry_logic():
    """
    Test that LLMClient correctly handles retries on server busy errors.
    """
    class RetryStrategy(LLMStrategy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.attempts = 0
            
        async def generate(self, input_data_string: str, system_instruction: str, response_schema: Type[BaseModel]) -> BaseModel:
            self.attempts += 1
            if self.attempts == 1:
                raise Exception("503 Server Busy")
            
            data = {}
            if "description" in response_schema.model_fields:
                data["description"] = "Success after retry"
            return response_schema(**data)

    strategy = RetryStrategy(api_key="test", model_name="test")
    client = LLMClient(strategy)
    
    class TestSchema(BaseModel):
        description: str
        
    # We need to mock asyncio.sleep to avoid waiting during tests
    with MagicMock() as mock_sleep:
        import asyncio
        original_sleep = asyncio.sleep
        
        # Mock sleep to return immediately
        async def mock_s(duration):
            return
        
        asyncio.sleep = mock_s
        
        response = await client.generate_description({"code": "code"}, "prompt", TestSchema)
        
        asyncio.sleep = original_sleep
    
    assert response.description == "Success after retry"
    assert strategy.attempts == 2
