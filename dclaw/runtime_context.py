from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .background import CronService, HeartbeatRunner
from .channels import ChannelManager
from .context import ContextGuard
from .delivery import DeliveryQueue, DeliveryRunner
from .gateway import GatewayServer
from .intelligence import SkillsManager
from .lanes import CommandQueue
from .runtime import AgentManager, BindingTable, SessionStore
from .terminal import BackgroundInbox


@dataclass
class RuntimeContext:
    mgr: ChannelManager
    bindings: BindingTable
    store: SessionStore
    agent_mgr: AgentManager
    guard: ContextGuard
    llm_client: Any
    command_queue: CommandQueue
    delivery_queue: DeliveryQueue
    delivery_runner: DeliveryRunner
    inbox: BackgroundInbox
    heartbeat: HeartbeatRunner
    cron_svc: CronService
    active_session_key: str
    bootstrap_data: dict[str, str]
    skills_mgr: SkillsManager
    skills_block: str
    force_agent_id: str | None = None
    gw_server: GatewayServer | None = None
    set_force_agent: Any = None
    set_gw_server: Any = None
