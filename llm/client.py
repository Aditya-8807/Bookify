import json
import re
from typing import Any, Dict

from openai import OpenAI
from anthropic import Anthropic

SUPPORTED_PROVIDERS = {"openai", "anthropic"}


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
        else:
            return Anthropic()

    def complete(self, system: str, user: str) -> str:
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
        else:
            resp = self._client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=8192,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        raw = self.complete(system=system, user=user)
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
        return json.loads(cleaned)


def client_from_config(config: dict) -> LLMClient:
    llm = config["llm"]
    return LLMClient(
        provider=llm["provider"],
        model=llm["model"],
        temperature=llm["temperature"],
    )
