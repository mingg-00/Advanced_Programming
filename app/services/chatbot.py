"""
Wrapper around the OpenAI Assistants API (or compatible endpoint).
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


class TradingAssistantChatbot:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        주식/투자 전용 한국어 챗봇 래퍼.

        - 항상 한국어로 답변
        - 개인 투자자에게 도움을 주는 \"트레이딩 어시스턴트\" 톤
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

        self.model = model
        self._client = OpenAI(api_key=self.api_key)

    def ask(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 한국어로만 답변하는 주식/투자 전문 어시스턴트입니다. "
                        "설명은 친절하고 간결하게, 개인 투자자가 이해하기 쉽게 해주세요. "
                        "코드나 숫자를 보여줄 때는 필요한 부분만 최소한으로 보여주세요."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response.choices[0].message.content or ""


