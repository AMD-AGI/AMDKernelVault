# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

from typing import List

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from models.Base import BaseModel


class GeminiModel(BaseModel):
    """Gemini API through Google's public OpenAI-compatible endpoint."""

    def __init__(self,
                 model_id="gemini-2.5-pro",
                 api_key=None):
        assert api_key is not None, "No API key provided."
        self.model_id = model_id
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self,
                 messages: List,
                 temperature=1.0,
                 presence_penalty=0,
                 frequency_penalty=0,
                 max_tokens=30000) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        if not response or not hasattr(response, "choices") or not response.choices:
            raise ValueError("The API returned no response choices.")
        return response.choices[0].message.content
