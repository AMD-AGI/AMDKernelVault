# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

from typing import List
import re
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from models.Base import BaseModel


class StandardOpenAIModel(BaseModel):
    """Standard OpenAI API (api.openai.com)"""
    def __init__(self, 
                 model_id="gpt-4o", 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        self.client = openai.OpenAI(api_key=api_key)
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List, 
                 temperature=0, 
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=5000) -> str:
        request = dict(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            n=1,
            stream=False,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        if re.fullmatch(r"gpt-5(?:-(?:mini|nano))?(?:-2025-\d{2}-\d{2})?", self.model_id):
            request["max_completion_tokens"] = request.pop("max_tokens")
            request.pop("temperature")
        response = self.client.chat.completions.create(**request)
        if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
            raise ValueError("No response choices returned from the API.")
        return response.choices[0].message.content


class OpenAIModel(StandardOpenAIModel):
    """Compatibility adapter for the public OpenAI API."""

    def __init__(self,
                 model_id="gpt-4o",
                 model_api_version="2024-06-01",
                 api_key=None):
        # Retain the legacy argument for callers. The public API does not use it.
        self.model_api_version = model_api_version
        super().__init__(model_id=model_id, api_key=api_key)

    def generate(self,
                 messages: List,
                 temperature=1.0,
                 presence_penalty=0,
                 frequency_penalty=0,
                 max_tokens=5000) -> str:
        return super().generate(
            messages=messages,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
        )
