import json
import os
import re
import threading
import time
from typing import Any, Dict

from openai import OpenAI, APIStatusError as OpenAIStatusError
from anthropic import Anthropic
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors

SUPPORTED_PROVIDERS = {"openai", "anthropic", "gemini"}


class TokenBudgetExceeded(Exception):
    pass

_RETRY_DELAYS = [15, 30, 60, 120, 180, 300]

# Per-provider pricing ($/1M tokens).
_PRICING = {
    "gemini":    {"input": 0.10,  "output": 0.40},  # gemini-2.0-flash
    "openai":    {"input": 2.50,  "output": 10.00},
    "anthropic": {"input": 3.00,  "output": 15.00},
}

_total_input_tokens  = 0
_total_output_tokens = 0
_active_provider     = "gemini"

# Rate limiter — enforces minimum gap between API calls.
_rate_lock           = threading.Lock()
_last_request_time   = 0.0
_min_request_interval = 0.0   # seconds; set by LLMClient.__init__


def get_cost_summary() -> str:
    pricing = _PRICING.get(_active_provider, {"input": 0, "output": 0})
    cost = (
        _total_input_tokens  / 1_000_000 * pricing["input"] +
        _total_output_tokens / 1_000_000 * pricing["output"]
    )
    return (
        f"Tokens used — input: {_total_input_tokens:,}  "
        f"output: {_total_output_tokens:,}  "
        f"estimated cost: ${cost:.4f}"
    )


def _enforce_rate_limit() -> None:
    global _last_request_time
    if _min_request_interval <= 0:
        return
    with _rate_lock:
        now = time.monotonic()
        wait = _min_request_interval - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (genai_errors.ServerError, genai_errors.ClientError)):
        return getattr(exc, "code", 0) in (429, 503)
    if isinstance(exc, OpenAIStatusError):
        return exc.status_code in (429, 503)
    return False


def _clean_json(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"//[^\n]*", "", text)
    return text.strip()


class LLMClient:
    def __init__(self, provider: str, model: str, temperature: float,
                 rate_limit_rpm: int = 0, api_key: str = None,
                 token_budget_usd: float = None):
        global _active_provider, _min_request_interval
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        self.provider = provider
        self.model = model
        self.temperature = temperature
        _active_provider = provider
        _min_request_interval = (60.0 / rate_limit_rpm) if rate_limit_rpm > 0 else 0.0
        self._api_key = api_key
        self._token_budget_usd = token_budget_usd
        self._client = self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            return OpenAI(api_key=self._api_key) if self._api_key else OpenAI()
        elif self.provider == "anthropic":
            return Anthropic(api_key=self._api_key) if self._api_key else Anthropic()
        else:  # gemini
            key = self._api_key or os.environ["GEMINI_API_KEY"]
            return genai.Client(api_key=key)

    def _call(self, system: str, user: str) -> str:
        global _total_input_tokens, _total_output_tokens

        _enforce_rate_limit()

        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            if resp.usage:
                _total_input_tokens  += resp.usage.prompt_tokens
                _total_output_tokens += resp.usage.completion_tokens
            return resp.choices[0].message.content

        elif self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=8192,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            if resp.usage:
                _total_input_tokens  += resp.usage.input_tokens
                _total_output_tokens += resp.usage.output_tokens
            return resp.content[0].text

        else:  # gemini
            resp = self._client.models.generate_content(
                model=self.model,
                contents=user,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self.temperature,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
            if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                _total_input_tokens  += getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
                _total_output_tokens += getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
            return resp.text

    def get_spend_usd(self) -> float:
        pricing = _PRICING.get(self.provider, {"input": 0.0, "output": 0.0})
        return (
            _total_input_tokens / 1_000_000 * pricing["input"] +
            _total_output_tokens / 1_000_000 * pricing["output"]
        )

    def _check_budget(self) -> None:
        if self._token_budget_usd and self.get_spend_usd() >= self._token_budget_usd:
            raise TokenBudgetExceeded(
                f"Token budget ${self._token_budget_usd:.2f} exceeded "
                f"(spent ${self.get_spend_usd():.4f})"
            )

    def complete(self, system: str, user: str) -> str:
        self._check_budget()
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


def client_from_config(config: dict, api_key: str = None) -> LLMClient:
    llm = config["llm"]
    return LLMClient(
        provider=llm["provider"],
        model=llm["model"],
        temperature=llm["temperature"],
        rate_limit_rpm=config.get("pipeline", {}).get("rate_limit_rpm", 0),
        api_key=api_key,
        token_budget_usd=config.get("pipeline", {}).get("token_budget_usd"),
    )
