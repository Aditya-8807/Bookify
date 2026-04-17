import json
import os
import re
import time
from typing import Any, Dict

from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors

SUPPORTED_PROVIDERS = {"openai", "anthropic", "gemini"}

_RETRY_DELAYS = [15, 30, 60, 120, 180, 300]

# Gemini 2.5 Flash pricing ($/1M tokens)
_GEMINI_INPUT_PRICE  = 0.15
_GEMINI_OUTPUT_PRICE = 0.60

_total_input_tokens  = 0
_total_output_tokens = 0


def get_cost_summary() -> str:
    cost = (_total_input_tokens  / 1_000_000 * _GEMINI_INPUT_PRICE +
            _total_output_tokens / 1_000_000 * _GEMINI_OUTPUT_PRICE)
    return (f"Tokens used — input: {_total_input_tokens:,}  "
            f"output: {_total_output_tokens:,}  "
            f"estimated cost: ${cost:.4f}")


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (genai_errors.ServerError, genai_errors.ClientError)):
        return getattr(exc, "status_code", 0) in (429, 503)
    return False


def _clean_json(raw: str) -> str:
    """Progressively strip common Gemini JSON formatting issues."""
    text = raw.strip()
    # Strip ```json ... ``` fences
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    # Strip trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Strip // line comments
    text = re.sub(r"//[^\n]*", "", text)
    return text.strip()


class LLMClient:
    def __init__(self, provider: str, model: str, temperature: float):
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self._client = self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            return OpenAI()
        elif self.provider == "anthropic":
            return Anthropic()
        else:
            return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _call(self, system: str, user: str) -> str:
        global _total_input_tokens, _total_output_tokens

        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content

        elif self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=8192,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

        else:  # gemini
            resp = self._client.models.generate_content(
                model=self.model,
                contents=user,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self.temperature,
                ),
            )
            if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                _total_input_tokens  += getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
                _total_output_tokens += getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
            return resp.text

    def complete(self, system: str, user: str) -> str:
        last_exc = None
        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if delay:
                print(f"[LLM] retrying in {delay}s (attempt {attempt})...", flush=True)
                time.sleep(delay)
            try:
                return self._call(system, user)
            except Exception as exc:
                if _is_retryable(exc):
                    last_exc = exc
                    continue
                raise
        raise last_exc

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        # Try up to 3 times: first with the raw response, then retry the LLM
        # with an explicit JSON reminder if parsing keeps failing.
        for attempt in range(3):
            if attempt == 0:
                raw = self.complete(system=system, user=user)
            else:
                print(f"[LLM] JSON parse failed, retrying with JSON reminder (attempt {attempt})...", flush=True)
                raw = self.complete(
                    system=system + "\n\nIMPORTANT: Return ONLY valid JSON. No markdown fences, no comments.",
                    user=user,
                )
            try:
                return json.loads(_clean_json(raw))
            except json.JSONDecodeError:
                continue
        raise ValueError(f"LLM returned invalid JSON after 3 attempts. Last response: {raw[:200]}")


def client_from_config(config: dict) -> LLMClient:
    llm = config["llm"]
    return LLMClient(
        provider=llm["provider"],
        model=llm["model"],
        temperature=llm["temperature"],
    )
