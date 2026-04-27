from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model_id: str
    dashscope_api_key: str
    dashscope_base_url: str | None
    anthropic_api_key: str
    anthropic_base_url: str | None
    openai_api_key: str
    openai_base_url: str | None


@dataclass(frozen=True)
class ChannelSettings:
    feishu_app_id: str
    feishu_app_secret: str
    feishu_encrypt_key: str
    feishu_bot_open_id: str
    feishu_is_lark: bool
    wechat_webhook_url: str


@dataclass(frozen=True)
class RuntimeSettings:
    heartbeat_interval: float
    heartbeat_active_start: int
    heartbeat_active_end: int
    resilience_max_retries: int
    resilience_circuit_threshold: int
    resilience_circuit_cooldown: float


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    workspace_dir: Path
    agents_dir: Path
    state_dir: Path
    cron_dir: Path
    delivery_dir: Path
    llm: LLMSettings
    channels: ChannelSettings
    runtime: RuntimeSettings


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env_str(name)
    if not value:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def load_config(env_path: Path = ENV_PATH, override: bool = True) -> AppConfig:
    load_dotenv(env_path, override=override)
    workspace_dir = PROJECT_ROOT / "workspace-main"
    state_dir = PROJECT_ROOT / ".state"
    return AppConfig(
        project_root=PROJECT_ROOT,
        workspace_dir=workspace_dir,
        agents_dir=PROJECT_ROOT / ".agents",
        state_dir=state_dir,
        cron_dir=workspace_dir / "cron",
        delivery_dir=workspace_dir / "delivery-queue",
        llm=LLMSettings(
            provider=_env_str("LLM_PROVIDER").lower(),
            model_id=_env_str("MODEL_ID", "MiniMax-M2.7"),
            dashscope_api_key=_env_str("DASHSCOPE_API_KEY"),
            dashscope_base_url=_env_str("DASHSCOPE_BASE_URL") or None,
            anthropic_api_key=_env_str("ANTHROPIC_API_KEY"),
            anthropic_base_url=_env_str("ANTHROPIC_BASE_URL") or None,
            openai_api_key=_env_str("OPENAI_API_KEY"),
            openai_base_url=_env_str("OPENAI_BASE_URL") or None,
        ),
        channels=ChannelSettings(
            feishu_app_id=_env_str("FEISHU_APP_ID"),
            feishu_app_secret=_env_str("FEISHU_APP_SECRET"),
            feishu_encrypt_key=_env_str("FEISHU_ENCRYPT_KEY"),
            feishu_bot_open_id=_env_str("FEISHU_BOT_OPEN_ID"),
            feishu_is_lark=_env_bool("FEISHU_IS_LARK"),
            wechat_webhook_url=_env_str("WECHAT_WEBHOOK_URL"),
        ),
        runtime=RuntimeSettings(
            heartbeat_interval=_env_float("HEARTBEAT_INTERVAL", 1800.0),
            heartbeat_active_start=_env_int("HEARTBEAT_ACTIVE_START", 9),
            heartbeat_active_end=_env_int("HEARTBEAT_ACTIVE_END", 22),
            resilience_max_retries=_env_int("RESILIENCE_MAX_RETRIES", 3),
            resilience_circuit_threshold=_env_int("RESILIENCE_CIRCUIT_THRESHOLD", 5),
            resilience_circuit_cooldown=_env_float("RESILIENCE_CIRCUIT_COOLDOWN", 300.0),
        ),
    )


CONFIG = load_config()
CONFIG.workspace_dir.mkdir(parents=True, exist_ok=True)
CONFIG.state_dir.mkdir(parents=True, exist_ok=True)

MODEL_ID = CONFIG.llm.model_id
