from __future__ import annotations

import json
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import dashscope

from .config import CONFIG

dashscope.base_http_api_url = CONFIG.llm.dashscope_base_url

STATE_DIR = CONFIG.state_dir
STATE_DIR.mkdir(parents=True, exist_ok=True)

class LLMClient(ABC):
    @abstractmethod
    def chat(
            self,
            model: str,
            system: str,
            messages: list[dict],
            max_tokens: int = 8096,
            tools: list[dict] | None = None,
             ) -> dict:
        pass

class FailoverReason(Enum):
    rate_limit = "rate_limit"
    auth = "auth"
    timeout = "timeout"
    billing = "billing"
    overflow = "overflow"
    network = "network"
    server = "server"
    unknown = "unknown"

def classify_failure(exc: Exception) -> FailoverReason:
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return FailoverReason.rate_limit
    if "401" in msg or "403" in msg or "auth" in msg or "api key" in msg or "invalid key" in msg:
        return FailoverReason.auth
    if "402" in msg or "billing" in msg or "quota" in msg or "insufficient" in msg:
        return FailoverReason.billing
    if "context" in msg or "token" in msg or "maximum context" in msg or "too long" in msg:
        return FailoverReason.overflow
    if "timeout" in msg or "timed out" in msg:
        return FailoverReason.timeout
    if "connection" in msg or "network" in msg or "dns" in msg or "tls" in msg:
        return FailoverReason.network
    if "500" in msg or "502" in msg or "503" in msg or "504" in msg or "server" in msg:
        return FailoverReason.server
    return FailoverReason.unknown

@dataclass
class ResilienceState:
    provider: str
    model: str
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    last_failure_reason: str = ""
    last_error: str = ""
    last_failure_at: float = 0.0
    last_success_at: float = 0.0
    total_attempts: int = 0
    total_successes: int = 0
    total_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "consecutive_failures": self.consecutive_failures,
            "cooldown_until": self.cooldown_until,
            "last_failure_reason": self.last_failure_reason,
            "last_error": self.last_error,
            "last_failure_at": self.last_failure_at,
            "last_success_at": self.last_success_at,
            "total_attempts": self.total_attempts,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ResilienceState":
        return ResilienceState(
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            consecutive_failures=int(data.get("consecutive_failures", 0)),
            cooldown_until=float(data.get("cooldown_until", 0.0)),
            last_failure_reason=str(data.get("last_failure_reason", "")),
            last_error=str(data.get("last_error", "")),
            last_failure_at=float(data.get("last_failure_at", 0.0)),
            last_success_at=float(data.get("last_success_at", 0.0)),
            total_attempts=int(data.get("total_attempts", 0)),
            total_successes=int(data.get("total_successes", 0)),
            total_failures=int(data.get("total_failures", 0)),
        )

class ResilienceManager:
    def __init__(
            self,
            state_path: Path,
            max_retries: int = 3,
            circuit_threshold: int = 5,
            circuit_cooldown: float = 300.0,
    ) -> None:
        self.state_path = state_path
        self.max_retries = max_retries
        self.circuit_threshold = circuit_threshold
        self.circuit_cooldown = circuit_cooldown
        self._lock = threading.Lock()
        self._states: dict[str, ResilienceState] = {}
        self._load()

    def _key(self, provider: str, model: str) -> str:
        return f"{provider}:{model}"

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._states = {
                key: ResilienceState.from_dict(value)
                for key, value in raw.get("states", {}).items()
                if isinstance(value, dict)
            }
        except (json.JSONDecodeError, OSError):
            self._states = {}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"states": {key: state.to_dict() for key, state in self._states.items()}}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            os.replace(str(tmp), str(self.state_path))
        except OSError:
            self.state_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_state(self, provider: str, model: str) -> ResilienceState:
        key = self._key(provider, model)
        with self._lock:
            if key not in self._states:
                self._states[key] = ResilienceState(provider=provider, model=model)
            return self._states[key]

    def _cooldown_for(self, reason: FailoverReason, failures: int) -> float:
        if reason == FailoverReason.auth:
            return self.circuit_cooldown
        if reason == FailoverReason.billing:
            return self.circuit_cooldown * 2
        if reason == FailoverReason.rate_limit:
            return min(600.0, 30.0 * max(1, failures))
        if reason in (FailoverReason.timeout, FailoverReason.network, FailoverReason.server):
            return min(180.0, 10.0 * max(1, failures))
        if reason == FailoverReason.overflow:
            return 0.0
        return min(120.0, 10.0 * max(1, failures))

    def _retry_delay(self, attempt: int, reason: FailoverReason) -> float:
        if reason in (FailoverReason.auth, FailoverReason.billing, FailoverReason.overflow):
            return 0.0
        base = min(8.0, 0.75 * (2 ** max(0, attempt - 1)))
        return base + random.uniform(0.0, base * 0.25)

    def run_chat(
            self,
            client: LLMClient,
            provider: str,
            model: str,
            **kwargs: Any,
    ) -> dict:
        state = self.get_state(provider, model)
        now = time.time()
        if state.cooldown_until > now:
            remaining = round(state.cooldown_until - now)
            raise RuntimeError(
                f"Resilience circuit open for {provider}:{model}; "
                f"retry in {remaining}s (reason={state.last_failure_reason})"
            )

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            with self._lock:
                state.total_attempts += 1
                self._save()
            try:
                response = client.chat(model=model, **kwargs)
                with self._lock:
                    state.consecutive_failures = 0
                    state.cooldown_until = 0.0
                    state.last_failure_reason = ""
                    state.last_error = ""
                    state.last_success_at = time.time()
                    state.total_successes += 1
                    self._save()
                return response
            except Exception as exc:
                last_exc = exc
                reason = classify_failure(exc)
                with self._lock:
                    state.consecutive_failures += 1
                    state.total_failures += 1
                    state.last_failure_reason = reason.value
                    state.last_error = str(exc)[:500]
                    state.last_failure_at = time.time()
                    cooldown = self._cooldown_for(reason, state.consecutive_failures)
                    if state.consecutive_failures >= self.circuit_threshold:
                        cooldown = max(cooldown, self.circuit_cooldown)
                    if cooldown > 0:
                        state.cooldown_until = max(state.cooldown_until, time.time() + cooldown)
                    self._save()
                if reason in (FailoverReason.auth, FailoverReason.billing, FailoverReason.overflow):
                    break
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt, reason))
        raise last_exc or RuntimeError("resilience run failed without exception")

    def status(self) -> list[dict[str, Any]]:
        now = time.time()
        with self._lock:
            rows = []
            for state in self._states.values():
                cooldown_remaining = max(0.0, state.cooldown_until - now)
                rows.append({
                    **state.to_dict(),
                    "cooldown_remaining": round(cooldown_remaining),
                    "available": cooldown_remaining <= 0,
                })
            return rows

# Anthropic Adapter
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, base_url: str | None = None):
        if not HAS_ANTHROPIC:
            raise RuntimeError("Please install anthropic: pip install anthropic")
        self.client = Anthropic(api_key=api_key, base_url=base_url)

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        converted = []
        for msg in messages:
            role = msg["role"]
            if role not in ("user", "assistant"):
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                converted.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}]
                })
            elif isinstance(content, list):
                anthropic_content = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            anthropic_content.append(block)
                        elif block.get("type") == "tool_result":
                            anthropic_content.append(block)
                        elif block.get("type") == "text":
                            anthropic_content.append(block)
                        else:
                            anthropic_content.append({"type": "text", "text": str(block)})
                    else:
                        anthropic_content.append({"type": "text", "text": str(block)})
                converted.append({
                    "role": role,
                    "content": anthropic_content or [{"type": "text", "text": ""}],
                })
        return converted

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
        return anthropic_tools

    def chat(self, model: str, system: str, messages: list[dict], max_tokens: int = 8096, tools: list[dict] | None = None) -> dict:
        converted_msgs = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        resp = self.client.messages.create(
            model=model,
            system=system,
            messages=converted_msgs,
            max_tokens=max_tokens,
            tools=anthropic_tools,
        )

        text_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })

        finish_reason = resp.stop_reason
        if finish_reason == "end_turn":
            finish_reason = "stop"
        elif finish_reason == "tool_use":
            finish_reason = "tool_calls"

        return {
            "text": "\n".join(text_parts),
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "raw": resp
        }

# OpenAI Adapter
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, base_url: str | None = None):
        if not HAS_OPENAI:
            raise RuntimeError("Please install openai: pip install openai")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, model: str, system: str, messages: list[dict], max_tokens: int = 8096, tools: list[dict] | None = None) -> dict:
        full_messages = [{"role": "system", "content": system}] + messages

        resp = self.client.chat.completions.create(
            model=model,
            messages=full_messages,
            max_tokens=max_tokens,
            tools=tools,
        )

        msg = resp.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })

        return {
            "text": msg.content or "",
            "tool_calls": tool_calls,
            "finish_reason": resp.choices[0].finish_reason,
            "raw": resp
        }

# DashScope Adapter
class DashScopeClient(LLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def chat(self, model: str, system: str, messages: list[dict], max_tokens: int = 8096, tools: list[dict] | None = None) -> dict:
        full_messages = [{"role": "system", "content": system}] + messages

        resp = dashscope.MultiModalConversation.call(
            api_key=self.api_key,
            model=model,
            messages=full_messages,
            max_tokens=max_tokens,
            result_format="message",
            tools=tools,
        )

        if resp.status_code != 200:
            raise Exception(f"DashScope error {resp.status_code}: {resp.message}")

        choice = resp.output.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason

        text = ""
        if hasattr(msg, "content"):
            if isinstance(msg.content, list):
                text = "\n".join(x.get("text", "") for x in msg.content if isinstance(x, dict))
            else:
                text = str(msg.content)

        tool_calls = []
        if finish_reason == "tool_calls" and hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"].get("arguments", "{}")
                    }
                })

        return {
            "text": text,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "raw": resp
        }

# LLM 工厂
class ResilientLLMClient(LLMClient):
    def __init__(self, inner: LLMClient, provider: str, resilience: ResilienceManager) -> None:
        self.inner = inner
        self.provider = provider
        self.resilience = resilience

    def chat(
            self,
            model: str,
            system: str,
            messages: list[dict],
            max_tokens: int = 8096,
            tools: list[dict] | None = None,
    ) -> dict:
        return self.resilience.run_chat(
            self.inner,
            provider=self.provider,
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools,
        )

RESILIENCE_MANAGER = ResilienceManager(
    STATE_DIR / "resilience.json",
    max_retries=CONFIG.runtime.resilience_max_retries,
    circuit_threshold=CONFIG.runtime.resilience_circuit_threshold,
    circuit_cooldown=CONFIG.runtime.resilience_circuit_cooldown,
)

def create_llm_client() -> LLMClient:
    settings = CONFIG.llm
    provider = settings.provider
    if not provider:
        if settings.anthropic_api_key:
            provider = "anthropic"
        elif settings.openai_api_key:
            provider = "openai"
        elif settings.dashscope_api_key:
            provider = "dashscope"
        else:
            raise RuntimeError("No LLM API key found. Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, or DASHSCOPE_API_KEY.")

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic provider.")
        return ResilientLLMClient(
            AnthropicClient(settings.anthropic_api_key, settings.anthropic_base_url),
            provider,
            RESILIENCE_MANAGER,
        )
    elif provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider.")
        return ResilientLLMClient(
            OpenAIClient(settings.openai_api_key, settings.openai_base_url),
            provider,
            RESILIENCE_MANAGER,
        )
    elif provider == "dashscope":
        if not settings.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for DashScope provider.")
        return ResilientLLMClient(
            DashScopeClient(settings.dashscope_api_key),
            provider,
            RESILIENCE_MANAGER,
        )
    else:
        raise RuntimeError(f"Unknown LLM provider: {provider}")
