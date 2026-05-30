"""Thin wrapper over OpenAI / Groq / Gemini.

Whichever API key is set wins. Order of preference: OpenAI, Groq, Gemini.
If none is set, callers should use the deterministic fallback in
nl_to_sql.py instead of calling get_client().
"""

from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from src.config import SETTINGS


class LLMUnavailable(RuntimeError):
    """No provider configured, or all providers errored out."""


class _BaseClient:
    name = "base"

    def chat(self, messages):
        raise NotImplementedError


class _OpenAIClient(_BaseClient):
    name = "openai"

    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(
            api_key=SETTINGS.openai_api_key,
            timeout=SETTINGS.llm_timeout_seconds,
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential_jitter(initial=1, max=4))
    def chat(self, messages):
        resp = self._client.chat.completions.create(
            model=SETTINGS.openai_model,
            messages=messages,
            temperature=SETTINGS.llm_temperature,
            max_tokens=SETTINGS.llm_max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


class _GroqClient(_BaseClient):
    name = "groq"

    def __init__(self):
        from groq import Groq
        self._client = Groq(
            api_key=SETTINGS.groq_api_key,
            timeout=SETTINGS.llm_timeout_seconds,
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential_jitter(initial=1, max=4))
    def chat(self, messages):
        resp = self._client.chat.completions.create(
            model=SETTINGS.groq_model,
            messages=messages,
            temperature=SETTINGS.llm_temperature,
            max_tokens=SETTINGS.llm_max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


class _GeminiClient(_BaseClient):
    name = "gemini"

    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=SETTINGS.gemini_api_key)
        self._model = genai.GenerativeModel(SETTINGS.gemini_model)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential_jitter(initial=1, max=4))
    def chat(self, messages):
        # Gemini has no system role - merge into the user prompt.
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


def get_client():
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


def active_provider():
    return SETTINGS.active_llm_provider
