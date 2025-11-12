"""
Wrapper around the OpenAI Assistants API (or compatible endpoint).
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


class TradingAssistantChatbot:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Set OPENAI_API_KEY environment variable or pass api_key explicitly."
            )
        self.model = model
        self._client = OpenAI(api_key=self.api_key)

    def ask(self, prompt: str) -> str:
        response = self._client.responses.create(
            model=self.model,
            input=prompt,
        )
        return response.output_text or ""


