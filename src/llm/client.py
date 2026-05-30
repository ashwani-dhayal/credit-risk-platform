"""Multi-provider LLM client with auto-detection.

Priority order (first non-empty key wins):
  1. OpenAI       (gpt-4o-mini default)
  2. Groq         (llama-3.1-8b-instant default, fast & free tier)
  3. Google Gemini (gemini-1.5-flash default, free tier)
  4. Fallback     (no network; rule-based intent parser in nl_to_sql)

Each provider is wrapped in a thin adapter exposing `.chat(messages)` that
returns a plain string. We use tenacity for one retry with exponential
back-off to handle transient 5xx/429 errors.
"""

from __future__ import annotations

from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from src.config import SETTINGS


class LLMUnavailable(RuntimeError):
    """Raised when no provider is configured or the call fails."""


class _BaseClient:
    name: str = "base"

    def chat(self, messages: list[dict]) -> str:
        raise NotImplementedError


class _OpenAIClient(_BaseClient):
    name = "openai"

    def __init__(self) -> None:
        from openai import OpenAI
        self._client = OpenAI(
            api_key=SETTINGS.openai_api_key,
            timeout=SETTINGS.llm_timeout_seconds,
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential_jitter(initial=1, max=4))
    def chat(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=SETTINGS.openai_model,
            messages=messages,
            temperature=SETTINGS.llm_temperature,
            max_tokens=SETTINGS.llm_max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


class _GroqClient(_BaseClient):
    name = "groq"

    def __init__(self) -> None:
        from groq import Groq
        self._client = Groq(
            api_key=SETTINGS.groq_api_key,
            timeout=SETTINGS.llm_timeout_seconds,
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential_jitter(initial=1, max=4))
    def chat(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=SETTINGS.groq_model,
            messages=messages,
            temperature=SETTINGS.llm_temperature,
            max_tokens=SETTINGS.llm_max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


class _GeminiClient(_BaseClient):
    name = "gemini"

    def __init__(self) -> None:
        import google.generativeai as genai
        genai.configure(api_key=SETTINGS.gemini_api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(SETTINGS.gemini_model)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential_jitter(initial=1, max=4))
    def chat(self, messages: list[dict]) -> str:
        # Gemini has no native "system" role — fold system into the first
        # user turn for parity with OpenAI/Groq.
        sys_parts = [m["content"] for m in messages if m["role"] == "system"]
        usr_parts = [m["content"] for m in messages if m["role"] != "system"]
        prompt = ("\n\n".join(sys_parts) + "\n\n" + "\n\n".join(usr_parts)).strip()
        resp = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": SETTINGS.llm_temperature,
                "max_output_tokens": SETTINGS.llm_max_tokens,
            },
        )
        return (getattr(resp, "text", None) or "").strip()


def get_client() -> _BaseClient:
    """Return the first available LLM client based on configured keys."""
    provider = SETTINGS.active_llm_provider
    if provider == "openai":
        return _OpenAIClient()
    if provider == "groq":
        return _GroqClient()
    if provider == "gemini":
        return _GeminiClient()
    raise LLMUnavailable(
        "No LLM API key is configured. Set OPENAI_API_KEY, GROQ_API_KEY, or "
        "GEMINI_API_KEY in your .env, or rely on the fallback parser."
    )


def active_provider() -> str:
    return SETTINGS.active_llm_provider
