"""Interchangeable LLM backends.

Both backends take the same messages plus the same JSON schema and return the
same raw JSON string, so nothing downstream of `complete()` knows or cares
which model produced an annotation.
"""

from __future__ import annotations

import json
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import requests

from .config import RunConfig

THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class BackendError(RuntimeError):
    """Raised when a backend cannot produce a usable response."""


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


def _strip_reasoning(text: str) -> str:
    """Remove <think> blocks that some local models emit before the JSON."""
    return THINK_TAG.sub("", text).strip()


def _extract_json(text: str) -> dict[str, Any]:
    """Parse the model's answer, tolerating fenced or prose-wrapped JSON."""
    cleaned = _strip_reasoning(text)
    if not cleaned:
        raise BackendError("model returned an empty response")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", cleaned, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise BackendError(f"could not parse JSON from response: {cleaned[:300]!r}")


class LLMBackend(ABC):
    """Common surface for every model provider."""

    name: str = "base"

    def __init__(self, config: RunConfig):
        self.config = config
        self.model = config.model

    @abstractmethod
    def _complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> LLMResponse:
        ...

    @abstractmethod
    def health_check(self) -> str:
        """Raise with an actionable message if the backend is unusable."""

    def complete_json(
        self, messages: list[dict[str, str]], schema: dict[str, Any]
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Call the model with retries and return parsed JSON plus metadata."""
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._complete(messages, schema)
                return _extract_json(response.text), response
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised
                last_error = exc
                if attempt == self.config.max_retries - 1:
                    break
                delay = self.config.retry_base_delay * (2**attempt)
                time.sleep(delay + random.uniform(0, 0.5 * delay))
        raise BackendError(f"{self.name}/{self.model} failed after {self.config.max_retries} attempts: {last_error}")


class OpenAIBackend(LLMBackend):
    """OpenAI Responses API with strict structured outputs (GPT-5.1 by default)."""

    name = "openai"

    def __init__(self, config: RunConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise BackendError("pip install openai") from exc
        if not config.openai_api_key:
            raise BackendError("OPENAI_API_KEY is not set")
        kwargs: dict[str, Any] = {"api_key": config.openai_api_key, "timeout": config.request_timeout}
        if config.openai_base_url:
            kwargs["base_url"] = config.openai_base_url
        self.client = OpenAI(**kwargs)

    @property
    def _is_reasoning_model(self) -> bool:
        return self.model.startswith(("gpt-5", "o1", "o3", "o4"))

    def health_check(self) -> str:
        try:
            self.client.models.retrieve(self.model)
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"cannot reach OpenAI model {self.model!r}: {exc}") from exc
        return f"openai:{self.model} ready"

    def _complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "max_output_tokens": self.config.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "thematic_annotation",
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        if self._is_reasoning_model:
            # GPT-5.1 supports "none" | "low" | "medium" | "high" and rejects
            # temperature; older reasoning models reject "none".
            payload["reasoning"] = {"effort": self.config.reasoning_effort}
            if self.config.reasoning_effort != "none":
                # Reasoning tokens are billed against max_output_tokens, so
                # without headroom the JSON gets truncated before it is emitted.
                payload["max_output_tokens"] = self.config.max_output_tokens + 4000
        else:
            payload["temperature"] = self.config.temperature

        started = time.perf_counter()
        response = self.client.responses.create(**payload)
        latency = time.perf_counter() - started

        if getattr(response, "status", "completed") == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", "unknown")
            raise BackendError(f"response incomplete ({reason}); raise max_output_tokens")

        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
            }
        return LLMResponse(text=response.output_text or "", latency_s=latency, usage=usage, raw=response)


class OllamaBackend(LLMBackend):
    """Local Ollama server (`ollama serve`) with JSON-schema constrained output."""

    name = "ollama"

    def __init__(self, config: RunConfig):
        super().__init__(config)
        self.host = config.ollama_host.rstrip("/")
        self._send_think = config.think is not None

    def health_check(self) -> str:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise BackendError(
                f"no Ollama server at {self.host} ({exc}). Start it with `ollama serve`."
            ) from exc
        available = [m["name"] for m in response.json().get("models", [])]
        if self.model not in available:
            raise BackendError(
                f"model {self.model!r} is not pulled. Available: {available}. "
                f"Run `ollama pull {self.model}`."
            )
        return f"ollama:{self.model} ready"

    def _post(self, body: dict[str, Any]) -> requests.Response:
        return requests.post(
            f"{self.host}/api/chat", json=body, timeout=self.config.request_timeout
        )

    def _complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_output_tokens,
                "seed": self.config.seed,
            },
        }
        if self._send_think:
            body["think"] = self.config.think

        started = time.perf_counter()
        response = self._post(body)
        if response.status_code == 400 and "think" in response.text.lower():
            # Model has no thinking mode; older servers reject the field itself.
            self._send_think = False
            body.pop("think", None)
            response = self._post(body)
        if response.status_code >= 400:
            raise BackendError(f"ollama HTTP {response.status_code}: {response.text[:300]}")
        latency = time.perf_counter() - started

        data = response.json()
        content = (data.get("message") or {}).get("content", "")
        usage = {
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
        }
        return LLMResponse(text=content, latency_s=latency, usage=usage, raw=data)


_BACKENDS: dict[str, type[LLMBackend]] = {"openai": OpenAIBackend, "ollama": OllamaBackend}


def get_backend(config: RunConfig) -> LLMBackend:
    try:
        backend_cls = _BACKENDS[config.backend]
    except KeyError as exc:
        raise BackendError(f"unknown backend {config.backend!r}") from exc
    return backend_cls(config)
