"""LLM synthesis client: abstract contract + OpenAI implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Contract for the final synthesis stage (grounded answer generation)."""

    @abstractmethod
    def synthesize(self, system_prompt: str, user_prompt: str) -> str:
        """Produce an answer given the strict system prompt and the
        context-augmented user prompt."""


class OpenAILLMClient(LLMClient):
    """OpenAI Chat Completions synthesizer (openai>=1.x SDK)."""

    def __init__(self, model: str, api_key: str | None) -> None:
        from openai import OpenAI  # deferred import keeps module import cheap

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def synthesize(self, system_prompt: str, user_prompt: str) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            # Near-zero temperature: synthesis should be deterministic
            # grounding over retrieved context, not creative generation.
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (completion.choices[0].message.content or "").strip()
