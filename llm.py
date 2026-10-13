"""LLM synthesis client: abstract contract + OpenAI implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMClient(ABC):
    """Contract for the final synthesis stage (grounded answer generation)."""

    @abstractmethod
    def synthesize(self, system_prompt: str, user_prompt: str) -> str:
        """Produce an answer given the strict system prompt and the
        context-augmented user prompt."""

    def synthesize_stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Stream an answer token-by-token.

        Default implementation falls back to the non-streaming `synthesize`
        so subclasses only need to implement one method.
        """
        yield self.synthesize(system_prompt, user_prompt)


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

    def synthesize_stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Yield answer deltas as the model produces them (server-sent streaming)."""
        stream = self._client.chat.completions.create(
            model=self._model,
            temperature=0.1,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        for chunk in stream:
            # Some chunks carry no choices (e.g. usage-only frames); skip them.
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
