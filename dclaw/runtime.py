from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_ID = os.getenv("MODEL_ID", "MiniMax-M2.7")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = PROJECT_ROOT / ".agents"
WORKDIR = PROJECT_ROOT

VALID_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
INVALID_CHARS_RE = re.compile(r"[^a-z0-9_-]+")
DEFAULT_AGENT_ID = "main"

def normalize_agent_id(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        return DEFAULT_AGENT_ID
    if VALID_ID_RE.match(trimmed):
        return trimmed.lower()
    cleaned = INVALID_CHARS_RE.sub("-", trimmed.lower().strip("-")[:64])
    return cleaned or DEFAULT_AGENT_ID

# ---------------------------------------------------------------------------
# 五层路由解析（peer_id, guild_id, account_id, channel, default）
# ---------------------------------------------------------------------------

@dataclass
class Binding:
    agent_id: str
    tier: int
    match_key: str # "peer_id" | "guild_id" | "account_id" | "channel" | "default"
    match_value: str
    priority: int = 0 # 大数优先

    def display(self) -> str:
        names = {1: "peer", 2: "guild", 3: "account", 4: "channel", 5: "default"}
        label = names.get(self.tier, f"tier-{self.tier}")
        return f"[{label}] {self.match_key}={self.match_value} -> agent:{self.agent_id} (pri={self.priority})"

class BindingTable:
    def __init__(self) -> None:
        self._bindings: list[Binding] = []

    def add(self, binding: Binding) -> None:
        self._bindings.append(binding)
        self._bindings.sort(key=lambda b: (b.tier, -b.priority))

    def remove(self, agent_id: str, match_key: str, match_value: str) -> bool:
        before = len(self._bindings)
        self._bindings = [
            b for b in self._bindings
            if not (b.agent_id == agent_id and b.match_key == match_key
                    and b.match_value == match_value)
        ]
        return len(self._bindings) < before

    def list_all(self) -> list[Binding]:
        return list(self._bindings)

    # 动态选择代理
    def resolve(self, channel: str = "", account_id: str = "",
                guild_id: str = "", peer_id: str = "") -> tuple[str | None, Binding | None]:
        for b in self._bindings:
            if b.tier == 1 and b.match_key == "peer_id":
                if ":" in b.match_value:
                    if b.match_value == f"{channel}:{peer_id}":
                        return b.agent_id, b
                elif b.match_value == peer_id:
                    return b.agent_id, b
            elif b.tier == 2 and b.match_key == "guild_id" and b.match_value == guild_id:
                return b.agent_id, b
            elif b.tier == 3 and b.match_key == "account_id" and b.match_value == account_id:
                return b.agent_id, b
            elif b.tier == 4 and b.match_key == "channel" and b.match_value == channel:
                return b.agent_id, b
            elif b.tier == 5 and b.match_key == "default":
                return b.agent_id, b
        return None, None

# ---------------------------------------------------------------------------
# 构建会话键 - dm_scope 控制隔离策略
# ---------------------------------------------------------------------------

def build_session_key(agent_id: str, channel: str = "", account_id: str = "",
                      peer_id: str = "", dm_scope: str = "per-peer") -> str:
    aid = normalize_agent_id(agent_id)
    ch = (channel or "unknown").strip().lower()
    acc = (account_id or "default").strip().lower()
    pid = (peer_id or "").strip().lower()
    if dm_scope == "per-account-channel-peer" and pid:
        return f"agent:{aid}:{ch}:{acc}:direct:{pid}"
    if dm_scope == "per-channel-peer" and pid:
        return f"agent:{aid}:{ch}:direct:{pid}"
    if dm_scope == "per-peer" and pid:
        return f"agent:{aid}:direct:{pid}"
    return f"agent:{aid}:main"

# ---------------------------------------------------------------------------
# Agent 配置和管理
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    id: str
    name: str
    personality: str = ""
    model: str = ""
    dm_scope: str = "per-peer"

    @property
    def effective_model(self) -> str:
        return self.model or MODEL_ID

    def system_prompt(self) -> str:
        parts = [f"You are {self.name}."]
        if self.personality:
            parts.append(f"Your personality: {self.personality}")
        parts.append("Answer questions helpfully and stay in character.")
        return " ".join(parts)

class AgentManager:
    def __init__(self, agents_base: Path | None = None, session_store: Any = None) -> None:
        self._agents: dict[str, AgentConfig] = {}
        self._agents_base = agents_base or AGENTS_DIR
        self._sessions: dict[str, list[dict]] = {}
        self._session_store = session_store
        self._lock = threading.RLock()

    def set_session_store(self, session_store: Any) -> None:
        with self._lock:
            self._session_store = session_store

    def register(self, config: AgentConfig) -> None:
        aid = normalize_agent_id(config.id)
        config.id = aid
        with self._lock:
            self._agents[aid] = config
        agent_dir = self._agents_base / aid
        (agent_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (WORKDIR / f"workspace-{aid}").mkdir(parents=True, exist_ok=True)

    def get_agent(self, agent_id: str) -> AgentConfig | None:
        with self._lock:
            return self._agents.get(normalize_agent_id(agent_id))

    def list_agents(self) -> list[AgentConfig]:
        with self._lock:
            return list(self._agents.values())

    def get_session(self, session_key: str) -> list[dict]:
        with self._lock:
            if session_key not in self._sessions:
                if self._session_store is not None:
                    self._sessions[session_key] = self._session_store.load_session(session_key)
                    self._session_store.ensure_session(session_key)
                else:
                    self._sessions[session_key] = []
            return self._sessions[session_key]

    def save_session(self, session_key: str) -> None:
        with self._lock:
            if self._session_store is not None and session_key in self._sessions:
                self._session_store.save_session(session_key, self._sessions[session_key])

    def list_sessions(self, agent_id: str = "") -> dict[str, int]:
        aid = normalize_agent_id(agent_id) if agent_id else ""
        with self._lock:
            return {k: len(v) for k, v in self._sessions.items()
                    if not aid or k.startswith(f"agent:{aid}:")}

class SessionStore:
    def __init__(self, agent_id: str = ""):
        self.agent_id = normalize_agent_id(agent_id) if agent_id else ""
        self.base_dir = WORKDIR / ".sessions" / "runtime"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "session.json"
        self._index: dict[str, dict] = self._load_index()
        self.current_session_id: str | None = None

    def _load_index(self) -> dict[str, dict]:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _file_stem(session_id: str) -> str:
        digest = hashlib.blake2b(session_id.encode("utf-8"), digest_size=8).hexdigest()
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", session_id).strip("-")[:80]
        return f"{slug or 'session'}-{digest}"

    def _session_path(self, session_id: str) -> Path:
        return self.base_dir / f"{self._file_stem(session_id)}.jsonl"

    @staticmethod
    def _parse_session_key(session_id: str) -> dict[str, str]:
        parts = session_id.split(":")
        meta = {"agent_id": "", "channel": "", "account_id": "", "peer_id": ""}
        if len(parts) >= 2 and parts[0] == "agent":
            meta["agent_id"] = parts[1]
        if len(parts) >= 3:
            meta["channel"] = parts[2]
        if "direct" in parts:
            idx = parts.index("direct")
            if idx + 1 < len(parts):
                meta["peer_id"] = parts[idx + 1]
            if idx >= 1:
                meta["account_id"] = parts[idx - 1] if idx >= 4 else ""
        return meta

    def ensure_session(self, session_id: str, label: str = "", metadata: dict[str, Any] | None = None) -> str:
        now = datetime.now(timezone.utc).isoformat()
        if session_id not in self._index:
            parsed = self._parse_session_key(session_id)
            parsed.update(metadata or {})
            self._index[session_id] = {
                "label": label,
                "session_key": session_id,
                "created_at": now,
                "last_active": now,
                "message_count": 0,
                **parsed,
            }
            self._save_index()
            self._session_path(session_id).touch()
        else:
            if label:
                self._index[session_id]["label"] = label
            if metadata:
                self._index[session_id].update(metadata)
            if label or metadata:
                self._save_index()
        self.current_session_id = session_id
        return session_id

    def create_session(
            self,
            label: str = "",
            session_key: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> str:
        session_id = session_key or f"manual:{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        self._index[session_id] = {
            "label": label,
            "session_key": session_id,
            "created_at": now,
            "last_active": now,
            "message_count": 0,
            **(metadata or self._parse_session_key(session_id)),
        }
        self._save_index()
        self._session_path(session_id).touch()
        self.current_session_id = session_id
        return session_id

    # 加载会话记录
    def load_session(self, session_id: str) -> list[dict]:
        path = self._session_path(session_id)
        if not path.exists():
            return []
        self.current_session_id = session_id
        return self._rebuild_history(path)

    def append_transcript(self, session_id: str, record: dict) -> None:
        self.ensure_session(session_id)
        path = self._session_path(session_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if session_id in self._index:
            self._index[session_id]["last_active"] = (
                datetime.now(timezone.utc).isoformat()
            )
            self._index[session_id]["message_count"] += 1
            self._save_index()

    def save_session(
            self,
            session_id: str,
            messages: list[dict],
            label: str = "",
            metadata: dict[str, Any] | None = None,
    ) -> None:
        self.ensure_session(session_id, label=label, metadata=metadata)
        path = self._session_path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._index[session_id]["last_active"] = datetime.now(timezone.utc).isoformat()
        self._index[session_id]["message_count"] = len(messages)
        self._save_index()

    # 根据 JSONL 重建 messages
    @staticmethod
    def _rebuild_history(path: Path) -> list[dict]:
        messages: list[dict] = []

        if not path.exists():
            return messages

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "role" in msg:
                messages.append(msg)
        return messages

    def list_sessions(self, agent_id: str = "") -> list[tuple[str, dict]]:
        items = list(self._index.items())
        if agent_id:
            aid = normalize_agent_id(agent_id)
            items = [(sid, meta) for sid, meta in items if meta.get("agent_id") == aid]
        items.sort(key=lambda x: x[1].get("last_active", ""), reverse=True)
        return items

# 精简消息格式 - LLM 生成摘要
def _serialize_messages_for_summary(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(f"[{role}]: {block.get('text', '')}")
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Agent 运行时
# ---------------------------------------------------------------------------

def setup_default_agent(agent_mgr: AgentManager, bindings: BindingTable) -> None:
    """初始化默认 Agent"""
    agent_mgr.register(AgentConfig(
        id="main",
        name="Main Agent",
        personality="",
    ))
    bindings.add(Binding(
        agent_id="main",
        tier=5,
        match_key="default",
        match_value="*",
    ))
