from __future__ import annotations
import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Type, Dict
from pydantic import BaseModel, Field
from loguru import logger

# from tostr.core.models import BaseStruct

class LLMResponse(BaseModel):
    uid: str = ""
    description: str = ""
    status: str = "success"
    error: str = ""
    child_descriptions: Dict["BaseStruct", str] = Field(default_factory=dict)

class LLMStrategy(ABC):
    def __init__(self, api_key: str, model_name: str, max_concurrent_requests: int = 200):
        self.api_key = api_key
        self.model_name = model_name
        self.max_concurrent_requests = max_concurrent_requests
        self._semaphore = None

    @property
    def semaphore(self):
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        return self._semaphore

    @abstractmethod
    async def generate(self, input_data_string: str, system_instruction: str, response_schema: Type[BaseModel]):
        pass

class LLMClient:
    def __init__(self, strategy: LLMStrategy, progress_tracker: "ProgressTracker" = None):
        self.strategy = strategy
        self.progress_tracker = progress_tracker

    async def generate_description(self, input_data: dict, system_prompt: str, response_schema: Type[BaseModel]) -> BaseModel | None:
        input_data_string = json.dumps(input_data)
        start_time = time.perf_counter()
        
        max_retries = 3
        base_delay = 2
        request_timeout = 10  # bounds a single attempt so one slow/stuck call can't stall the
        # whole describe tree — a directory's own LLM call only fires after every descendant's
        # call has finished (see _describe_directory), so an unbounded hang here blocks its
        # entire ancestor chain, not just this one struct.

        async with self.strategy.semaphore:
            for attempt in range(max_retries):
                try:
                    result = await asyncio.wait_for(
                        self.strategy.generate(
                            input_data_string=input_data_string,
                            system_instruction=system_prompt,
                            response_schema=response_schema
                        ),
                        timeout=request_timeout,
                    )

                    elapsed_time = time.perf_counter() - start_time
                    logger.debug(f"LLM Generation successful in {elapsed_time:.4f} seconds")

                    if self.progress_tracker:
                        # Advance once for the primary struct being processed in this call
                        self.progress_tracker.advance('describe', 1)

                    return result

                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        sleep_time = base_delay * (2 ** attempt)
                        logger.warning(f"⏳ LLM call exceeded {request_timeout}s. Retrying in {sleep_time}s...")
                        await asyncio.sleep(sleep_time)
                        continue
                    logger.error(f"LLM Generation failed: timed out after {request_timeout}s on all {max_retries} attempts")
                    return None

                except Exception as e:
                    error_str = str(e)
                    if "503" in error_str or "429" in error_str:
                        if attempt < max_retries - 1:
                            sleep_time = base_delay * (2 ** attempt)
                            logger.warning(f"⏳ LLM Server busy. Retrying in {sleep_time}s...")
                            await asyncio.sleep(sleep_time)
                            continue

                    logger.error(f"LLM Generation failed: {error_str}")
                    return None