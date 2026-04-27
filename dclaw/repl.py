from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .background import CronService, HeartbeatRunner, get_event_loop, run_async
from .channels import ChannelManager
from .context import ContextGuard
from .delivery import DeliveryQueue, DeliveryRunner
from .engine import resolve_route, run_agent
from .gateway import GatewayServer
from .intelligence import SkillsManager, build_system_prompt
from .lanes import CommandQueue
from .llm import LLMClient, RESILIENCE_MANAGER
from .runtime import (
    AgentManager,
    BindingTable,
    DEFAULT_AGENT_ID,
    SessionStore,
    normalize_agent_id,
)
from .terminal import (
    BLUE,
    BOLD,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    RESET,
    YELLOW,
    BackgroundInbox,
    print_channel,
    print_cron,
    print_heartbeat,
    print_info,
    print_section,
    print_session,
    print_warn,
)
from .tools import get_tool_policy, memory_store, set_tool_mode

# 工作目录
WORKDIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = WORKDIR / "workspace-main"

def _default_auto_recall(user_message: str, top_k: int = 3) -> str:
    return ""

def _default_refresh_intelligence(mode: str = "full") -> None:
    return None

def _empty_bootstrap() -> dict[str, str]:
    return {}

def _empty_skills_block() -> str:
    return ""

def _empty_skills_manager() -> SkillsManager:
    return SkillsManager(WORKSPACE_DIR)

_auto_recall = _default_auto_recall
_refresh_intelligence = _default_refresh_intelligence
_get_bootstrap_data = _empty_bootstrap
_get_skills_block = _empty_skills_block
_get_skills_manager = _empty_skills_manager


@dataclass
class ReplServices:
    auto_recall: Any = _default_auto_recall
    refresh_intelligence: Any = _default_refresh_intelligence
    get_bootstrap_data: Any = _empty_bootstrap
    get_skills_block: Any = _empty_skills_block
    get_skills_manager: Any = _empty_skills_manager
    run_agent: Any = run_agent
    resolve_route: Any = resolve_route


DEFAULT_REPL_SERVICES = ReplServices()

def format_memory_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    return "\n".join(
        f"- [{item['path']}] (score: {item['score']}) {item['snippet']}"
        for item in results
    )

def configure_repl(
        auto_recall_func=None,
        refresh_intelligence_func=None,
        get_bootstrap_data_func=None,
        get_skills_block_func=None,
        get_skills_manager_func=None,
) -> None:
    global _auto_recall, _refresh_intelligence, _get_bootstrap_data, _get_skills_block, _get_skills_manager
    if auto_recall_func is not None:
        _auto_recall = auto_recall_func
    if refresh_intelligence_func is not None:
        _refresh_intelligence = refresh_intelligence_func
    if get_bootstrap_data_func is not None:
        _get_bootstrap_data = get_bootstrap_data_func
    if get_skills_block_func is not None:
        _get_skills_block = get_skills_block_func
    if get_skills_manager_func is not None:
        _get_skills_manager = get_skills_manager_func

def get_repl_services() -> ReplServices:
    return ReplServices(
        auto_recall=_auto_recall,
        refresh_intelligence=_refresh_intelligence,
        get_bootstrap_data=_get_bootstrap_data,
        get_skills_block=_get_skills_block,
        get_skills_manager=_get_skills_manager,
    )

# ---------------------------------------------------------------------------
# Read-Eval-Print Loop
# ---------------------------------------------------------------------------

def handle_repl_command(
        command: str,
        store: SessionStore,
        guard: ContextGuard,
        messages: list[dict],
        mgr: ChannelManager,
        llm_client: LLMClient,
        model_id: str,
        bindings: BindingTable | None = None,
        agent_mgr: AgentManager | None = None,
        force_agent_id: Optional[str] = None,
        set_force_agent: Optional[Callable[..., Any]] = None,
        gw_server: Optional["GatewayServer"] = None,
        set_gw_server: Optional[Callable[..., Any]] = None,
        heartbeat: Optional[HeartbeatRunner] = None,
        cron_svc: Optional[CronService] = None,
        delivery_queue: Optional[DeliveryQueue] = None,
        delivery_runner: Optional[DeliveryRunner] = None,
        inbox: Optional[BackgroundInbox] = None,
        command_queue: Optional[CommandQueue] = None,
        active_session_key: str = "",
        bootstrap_data: dict[str, str] | None = None,
        skills_mgr: SkillsManager | None = None,
        skills_block: str = "",
        runtime_context: Any | None = None,
        services: ReplServices | None = None,
) -> tuple[bool, list[dict], Optional[str], Optional["GatewayServer"]]:
    svc = services or get_repl_services()
    if runtime_context is not None:
        bindings = bindings or runtime_context.bindings
        agent_mgr = agent_mgr or runtime_context.agent_mgr
        force_agent_id = runtime_context.force_agent_id if force_agent_id is None else force_agent_id
        set_force_agent = set_force_agent or runtime_context.set_force_agent
        gw_server = runtime_context.gw_server if gw_server is None else gw_server
        set_gw_server = set_gw_server or runtime_context.set_gw_server
        heartbeat = heartbeat or runtime_context.heartbeat
        cron_svc = cron_svc or runtime_context.cron_svc
        delivery_queue = delivery_queue or runtime_context.delivery_queue
        delivery_runner = delivery_runner or runtime_context.delivery_runner
        inbox = inbox or runtime_context.inbox
        command_queue = command_queue or runtime_context.command_queue
        active_session_key = active_session_key or runtime_context.active_session_key
        bootstrap_data = bootstrap_data or runtime_context.bootstrap_data
        skills_mgr = skills_mgr or runtime_context.skills_mgr
        skills_block = skills_block or runtime_context.skills_block

    parts = command.strip().split(maxsplit = 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/new":
        label = arg or ""
        sid = store.create_session(label=label or "cli")
        if runtime_context is not None:
            runtime_context.active_session_key = sid
        print_session(f"Created new session: {sid}" + (f" ({label})" if label else ""))
        return True, [], force_agent_id, gw_server
    elif cmd == "/list":
        sessions = store.list_sessions()
        if not sessions:
            print_info("No sessions found.")
            return True, messages, force_agent_id, gw_server

        print_info("Sessions:")
        for sid, meta in sessions:
            active = "<-- current" if sid == store.current_session_id else ""
            label = meta.get("label", "")
            label_str = f" {label}" if label else ""
            count = meta.get("message_count", 0)
            last = meta.get("last_active", "?")[:19]
            print_info(
                f" {sid}{label_str} "
                f"msgs={count} last={last}{active}"
            )
        return True, messages, force_agent_id, gw_server

    elif cmd == "/switch":
        if not arg:
            print_warn("Usage: /switch <session_id>")
            return True, messages, force_agent_id, gw_server
        query = arg.strip()

        # 会话ID匹配
        matched = [
            sid for sid in store._index if sid.startswith(query)
        ]

        # 会话名称匹配
        if not matched:
            matched = [
                sid for sid, meta in store._index.items()
                if meta.get("label").startswith(query)
            ]

        if len(matched) == 0:
            print_warn(f"Session not found: {query}")
            return True, messages, force_agent_id, gw_server
        if len(matched) > 1:
            print_warn(f"Ambiguous prefix, matches: {', '.join(matched)}")
            return True, messages, force_agent_id, gw_server

        new_sid = matched[0]
        new_messages = store.load_session(new_sid)
        if runtime_context is not None:
            runtime_context.active_session_key = new_sid
        print_session(f" Switched to session: {new_sid} ({len(new_messages)} messages)")
        return True, new_messages, force_agent_id, gw_server

    elif cmd == "/context":
        estimated = guard.estimate_messages_tokens(messages)
        pct = (estimated / guard.max_tokens) * 100
        bar_len = 30
        filled = int(bar_len * min(pct, 100) / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        color = GREEN if pct < 50 else (YELLOW if pct < 80 else RED)
        print_info(f"  Context usage: ~{estimated:,} / {guard.max_tokens:,} tokens")
        print(f"  {color}[{bar}] {pct:.1f}%{RESET}")
        print_info(f"Messages: {len(messages)}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/compact":
        if len(messages) <= 2:
            print_info("Too few messages to compact (need > 2).")
            return True, messages, force_agent_id, gw_server
        print_session("Compacting history...")
        new_messages = guard.compact_history(messages, llm_client, model_id)
        print_session(f"{len(messages)} -> {len(new_messages)} messages")
        return True, new_messages, force_agent_id, gw_server

    elif cmd == "/channels":
        channels = mgr.list_channels()
        if channels:
            print_channel("Channels:")
            for name in channels:
                print_channel(f"  - {name}")
        else:
            print_info("No channels.")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/soul":
        print_section("SOUL.md")
        soul = (bootstrap_data or {}).get("SOUL.md", "")
        print(soul if soul else f"{DIM}(SOUL.md not found in {WORKSPACE_DIR}){RESET}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/skills":
        mgr_skills = skills_mgr or svc.get_skills_manager()
        print_section("Skills")
        if not mgr_skills.skills:
            print(f"{DIM}(no skills discovered){RESET}")
        else:
            for skill in mgr_skills.skills:
                print(f"- {skill['name']}: {skill.get('description', '')}")
                print(f"  path={skill.get('path', '')}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/memory":
        print_section("Memory")
        stats = memory_store.get_stats()
        print(f"  MEMORY.md chars: {stats['evergreen_chars']}")
        print(f"  daily files: {stats['daily_files']}")
        print(f"  daily entries: {stats['daily_entries']}")
        print(f"  dir: {memory_store.memory_dir}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/search":
        if not arg:
            print_warn("Usage: /search <query>")
            return True, messages, force_agent_id, gw_server
        print_section(f"Memory Search: {arg}")
        results = memory_store.hybrid_search(arg)
        if not results:
            print(f"{DIM}(no relevant memories){RESET}")
        else:
            print(format_memory_results(results))
        return True, messages, force_agent_id, gw_server

    elif cmd == "/prompt":
        prompt = build_system_prompt(
            mode="full",
            bootstrap=bootstrap_data or svc.get_bootstrap_data(),
            skills_block=skills_block or svc.get_skills_block(),
            memory_context=svc.auto_recall("show prompt"),
            agent_id=force_agent_id or DEFAULT_AGENT_ID,
            channel="cli",
        )
        print_section("System Prompt")
        if len(prompt) > 3000:
            print(prompt[:3000])
            print(f"\n{DIM}... ({len(prompt) - 3000} more chars, total {len(prompt)}){RESET}")
        else:
            print(prompt)
        return True, messages, force_agent_id, gw_server

    elif cmd == "/bootstrap":
        print_section("Bootstrap")
        data = bootstrap_data or svc.get_bootstrap_data()
        if not data:
            print(f"{DIM}(no bootstrap files loaded from {WORKSPACE_DIR}){RESET}")
        else:
            for name, content in data.items():
                print(f"- {name}: {len(content)} chars")
            print(f"  total: {sum(len(v) for v in data.values())} chars")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/reload_intelligence":
        svc.refresh_intelligence()
        print_info(
            f"Reloaded intelligence: bootstrap={len(svc.get_bootstrap_data())} files, "
            f"skills={len(svc.get_skills_manager().skills)}"
        )
        return True, messages, force_agent_id, gw_server

    elif cmd == "/tool-mode":
        if not arg:
            policy = get_tool_policy()
            print_info(f"  Tool mode: {policy.mode.value}")
            print_info("  Available modes: safe, dev, trusted")
            return True, messages, force_agent_id, gw_server
        try:
            mode = set_tool_mode(arg)
        except ValueError as exc:
            print_warn(str(exc))
            return True, messages, force_agent_id, gw_server
        print_info(f"  Tool mode -> {mode.value}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/heartbeat":
        if heartbeat is None:
            print_warn("Heartbeat not available")
            return True, messages, force_agent_id, gw_server
        for key, value in heartbeat.status().items():
            print_info(f"  {key}: {value}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/trigger":
        if heartbeat is None:
            print_warn("Heartbeat not available")
            return True, messages, force_agent_id, gw_server
        print_info(f"  {heartbeat.trigger()}")
        for item in heartbeat.drain_output():
            print_heartbeat(item)
        return True, messages, force_agent_id, gw_server

    elif cmd == "/cron":
        if cron_svc is None:
            print_warn("Cron not available")
            return True, messages, force_agent_id, gw_server
        status = cron_svc.status()
        print_info(f"  CRON.json: {status['cron_file']}")
        print_info(f"  croniter: {'yes' if status['croniter'] else 'no (using built-in 5-field fallback)'}")
        if status["running_job_id"]:
            print_info(f"  running: {status['running_job_id']}")
        jobs = cron_svc.list_jobs()
        if not jobs:
            print_info("  No cron jobs.")
        for job in jobs:
            tag = f"{GREEN}ON{RESET}" if job["enabled"] else f"{RED}OFF{RESET}"
            err = f" {YELLOW}err:{job['errors']}{RESET}" if job["errors"] else ""
            nxt = f" in {job['next_in']}s" if job["next_in"] is not None else " unscheduled"
            print(f"  [{tag}] {job['id']} - {job['name']} ({job['kind']}){err}{nxt}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/cron-trigger":
        if cron_svc is None:
            print_warn("Cron not available")
            return True, messages, force_agent_id, gw_server
        if not arg:
            print_warn("Usage: /cron-trigger <job_id>")
            return True, messages, force_agent_id, gw_server
        print_info(f"  {cron_svc.trigger_job(arg)}")
        for item in cron_svc.drain_output():
            print_cron(item)
        return True, messages, force_agent_id, gw_server

    elif cmd == "/cron-reload":
        if cron_svc is None:
            print_warn("Cron not available")
            return True, messages, force_agent_id, gw_server
        print_info(f"  {cron_svc.reload()}")
        for item in cron_svc.drain_output():
            print_cron(item)
        return True, messages, force_agent_id, gw_server

    elif cmd == "/delivery":
        if delivery_queue is None or delivery_runner is None:
            print_warn("Delivery not available")
            return True, messages, force_agent_id, gw_server
        stats = delivery_runner.get_stats()
        print_info(f"  queue: {delivery_queue.queue_dir}")
        print_info(
            f"  pending={stats['pending']} failed={stats['failed']} "
            f"attempted={stats['total_attempted']} "
            f"succeeded={stats['total_succeeded']} errors={stats['total_failed']}"
        )
        return True, messages, force_agent_id, gw_server

    elif cmd == "/queue":
        if delivery_queue is None:
            print_warn("Delivery not available")
            return True, messages, force_agent_id, gw_server
        pending = delivery_queue.load_pending()
        if not pending:
            print_info("  Delivery queue is empty.")
            return True, messages, force_agent_id, gw_server
        now = time.time()
        print_info(f"  Pending deliveries ({len(pending)}):")
        for entry in pending:
            wait = f" wait={entry.next_retry_at - now:.0f}s" if entry.next_retry_at > now else ""
            preview = entry.text[:50].replace("\n", " ")
            print_info(
                f"    {entry.id} {entry.channel}:{entry.to} "
                f"retry={entry.retry_count}{wait} \"{preview}\""
            )
        return True, messages, force_agent_id, gw_server

    elif cmd == "/failed":
        if delivery_queue is None:
            print_warn("Delivery not available")
            return True, messages, force_agent_id, gw_server
        failed = delivery_queue.load_failed()
        if not failed:
            print_info("  No failed deliveries.")
            return True, messages, force_agent_id, gw_server
        print_info(f"  Failed deliveries ({len(failed)}):")
        for entry in failed:
            preview = entry.text[:50].replace("\n", " ")
            print_info(
                f"    {entry.id} {entry.channel}:{entry.to} "
                f"retries={entry.retry_count} error=\"{entry.last_error[:40]}\" "
                f"\"{preview}\""
            )
        return True, messages, force_agent_id, gw_server

    elif cmd == "/retry-failed":
        if delivery_queue is None:
            print_warn("Delivery not available")
            return True, messages, force_agent_id, gw_server
        print_info(f"  Moved {delivery_queue.retry_failed()} failed deliveries back to queue.")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/inbox":
        if inbox is None:
            print_warn("Inbox not available")
            return True, messages, force_agent_id, gw_server
        if arg.strip().lower() == "clear":
            print_info(f"  Cleared {inbox.clear()} inbox item(s).")
            return True, messages, force_agent_id, gw_server
        items = inbox.list_items()
        if not items:
            print_info("  Inbox is empty.")
            return True, messages, force_agent_id, gw_server
        print_section(f"Background Inbox ({len(items)})")
        for idx, item in enumerate(items, 1):
            ts = item.get("ts", "")[11:19] or "??:??:??"
            source = item.get("source", "background")
            text = item.get("text", "")
            print(f"{DIM}[{idx}] {ts} {source}{RESET}")
            print(text)
            print()
        print_info("  Use /inbox clear to clear these items.")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/resilience":
        rows = RESILIENCE_MANAGER.status()
        if not rows:
            print_info("  No resilience state recorded yet.")
            return True, messages, force_agent_id, gw_server
        print_section("Resilience")
        for row in rows:
            status = "available" if row["available"] else f"cooldown {row['cooldown_remaining']}s"
            last_success = (
                datetime.fromtimestamp(row["last_success_at"]).isoformat()
                if row["last_success_at"] else "never"
            )
            print_info(
                f"  {row['provider']}:{row['model']} {status} "
                f"failures={row['consecutive_failures']} "
                f"success={row['total_successes']}/{row['total_attempts']} "
                f"last_success={last_success}"
            )
            if row.get("last_failure_reason"):
                print_info(f"    last_failure={row['last_failure_reason']} {row.get('last_error', '')[:120]}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/lanes":
        if command_queue is None:
            print_warn("Command queue not available")
            return True, messages, force_agent_id, gw_server
        stats = command_queue.stats()
        if not stats:
            print_info("  No lanes.")
            return True, messages, force_agent_id, gw_server
        for name, row in stats.items():
            active_bar = "*" * row["active"] + "." * max(0, row["max_concurrency"] - row["active"])
            print_info(
                f"  {name:12s} active=[{active_bar}] "
                f"queued={row['queue_depth']} max={row['max_concurrency']} "
                f"generation={row['generation']}"
            )
        return True, messages, force_agent_id, gw_server

    elif cmd == "/lane-queue":
        if command_queue is None:
            print_warn("Command queue not available")
            return True, messages, force_agent_id, gw_server
        stats = command_queue.stats()
        busy = {name: row for name, row in stats.items() if row["active"] or row["queue_depth"]}
        if not busy:
            print_info("  All lanes are idle.")
            return True, messages, force_agent_id, gw_server
        for name, row in busy.items():
            print_info(f"  {name}: active={row['active']} queued={row['queue_depth']}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/concurrency":
        if command_queue is None:
            print_warn("Command queue not available")
            return True, messages, force_agent_id, gw_server
        parts2 = arg.strip().split()
        if len(parts2) != 2:
            print_warn("Usage: /concurrency <lane> <N>")
            return True, messages, force_agent_id, gw_server
        try:
            new_max = max(1, int(parts2[1]))
        except ValueError:
            print_warn("N must be an integer")
            return True, messages, force_agent_id, gw_server
        command_queue.set_max_concurrency(parts2[0], new_max)
        print_info(f"  {parts2[0]} max_concurrency -> {new_max}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/generation":
        if command_queue is None:
            print_warn("Command queue not available")
            return True, messages, force_agent_id, gw_server
        for name, row in command_queue.stats().items():
            print_info(f"  {name}: generation={row['generation']}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/reset-lanes":
        if command_queue is None:
            print_warn("Command queue not available")
            return True, messages, force_agent_id, gw_server
        generations = command_queue.reset_all()
        if not generations:
            print_info("  No lanes to reset.")
        else:
            for name, generation in generations.items():
                print_info(f"  {name}: generation -> {generation}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/accounts":
        accounts = mgr.accounts
        if accounts:
            print_channel("Accounts:")
            for acc in accounts:
                masked = acc.token[:8] + "..." if len(acc.token) > 8 else "(none)"
                print_channel(f"- {acc.channel}/{acc.account_id}  token={masked}")
        else:
            print_info("No accounts.")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/bindings":
        if bindings is None:
            print_warn("Bindings not available")
            return True, messages, force_agent_id, gw_server
        all_b = bindings.list_all()
        if not all_b:
            print_info("(no bindings)")
        else:
            print(f"\n{BOLD}Route Bindings ({len(all_b)}):{RESET}")
            for b in all_b:
                c = [MAGENTA, BLUE, CYAN, GREEN, DIM][min(b.tier - 1, 4)]
                print(f"  {c}{b.display()}{RESET}")
            print()
        return True, messages, force_agent_id, gw_server

    elif cmd == "/route":
        if bindings is None or agent_mgr is None:
            print_warn("Bindings or AgentManager not available")
            return True, messages, force_agent_id, gw_server
        parts = arg.strip().split()
        if len(parts) < 2:
            print_warn("Usage: /route <channel> <peer_id> [account_id] [guild_id]")
            return True, messages, force_agent_id, gw_server
        ch, pid = parts[0], parts[1]
        acc = parts[2] if len(parts) > 2 else ""
        gid = parts[3] if len(parts) > 3 else ""
        aid, sk = resolve_route(bindings, agent_mgr, channel=ch, peer_id=pid, account_id=acc, guild_id=gid)
        a = agent_mgr.get_agent(aid)
        print(f"\n{BOLD}Route Resolution:{RESET}")
        print(f"  {DIM}Input:   ch={ch} peer={pid} acc={acc or '-'} guild={gid or '-'}{RESET}")
        print(f"  {CYAN}Agent:   {aid} ({a.name if a else '?'}){RESET}")
        print(f"  {GREEN}Session: {sk}{RESET}\n")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/agents":
        if agent_mgr is None:
            print_warn("AgentManager not available")
            return True, messages, force_agent_id, gw_server
        agents = agent_mgr.list_agents()
        if not agents:
            print_info("(no agents)")
        else:
            print(f"\n{BOLD}Agents ({len(agents)}):{RESET}")
            for a in agents:
                print(f"  {CYAN}{a.id}{RESET} ({a.name})  model={a.effective_model}  dm_scope={a.dm_scope}")
                if a.personality:
                    print(f"    {DIM}{a.personality[:70]}{'...' if len(a.personality) > 70 else ''}{RESET}")
            print()
        return True, messages, force_agent_id, gw_server

    elif cmd == "/switch_agent":
        if set_force_agent is None:
            print_warn("Agent switching not available")
            return True, messages, force_agent_id, gw_server
        if not arg:
            print_info(f"force_agent: {force_agent_id or '(off)'}")
            return True, messages, force_agent_id, gw_server
        if arg.lower() == "off":
            set_force_agent(None)
            force_agent_id = None
            print_info("Routing mode restored.")
        else:
            if agent_mgr is not None and agent_mgr.get_agent(arg):
                force_agent_id = normalize_agent_id(arg)
                set_force_agent(force_agent_id)
                print_info(f"Forcing agent: {force_agent_id}")
            else:
                print_warn(f"Agent not found: {arg}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/gateway":
        if set_gw_server is None:
            print_warn("Gateway not available")
            return True, messages, force_agent_id, gw_server
        if arg == "start":
            if gw_server is None:
                if agent_mgr is None or bindings is None:
                    print_warn("AgentManager or Bindings not available")
                    return True, messages, force_agent_id, gw_server
                new_gw = GatewayServer(
                    agent_mgr,
                    bindings,
                    llm_client=llm_client,
                    command_queue=command_queue,
                    run_agent_func=svc.run_agent,
                    run_async_func=run_async,
                    resolve_route_func=svc.resolve_route,
                )
                asyncio.run_coroutine_threadsafe(new_gw.start(), get_event_loop()).result()
                set_gw_server(new_gw)
                gw_server = new_gw
                print_info("Gateway started: ws://localhost:8765")
            else:
                print_info("Gateway already running")
        elif arg == "stop":
            if gw_server is not None:
                asyncio.run_coroutine_threadsafe(gw_server.stop(), get_event_loop()).result()
                set_gw_server(None)
                gw_server = None
                print_info("Gateway stopped")
            else:
                print_info("Gateway not running")
        elif arg == "status":
            if gw_server:
                print_info(f"Gateway running: ws://localhost:8765")
                print_info(f"  Agents: {len(agent_mgr.list_agents()) if agent_mgr else 0}")
                print_info(f"  Bindings: {len(bindings.list_all()) if bindings else 0}")
            else:
                print_info("Gateway not running")
        else:
            print_info("Usage: /gateway [start|stop|status]")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/sessions":
        if agent_mgr is None:
            print_warn("AgentManager not available")
            return True, messages, force_agent_id, gw_server
        s = agent_mgr.list_sessions()
        if not s:
            print_info("(no sessions)")
        else:
            print(f"\n{BOLD}Sessions ({len(s)}):{RESET}")
            for k, n in sorted(s.items()):
                print(f"  {GREEN}{k}{RESET} ({n} msgs)")
            print()
        return True, messages, force_agent_id, gw_server

    elif cmd == "/help":
        print_info("  Session Commands:")
        print_info("    /new [label]       Create a new session")
        print_info("    /list              List all sessions")
        print_info("    /switch <id>       Switch to a session (prefix match)")
        print_info("    /context           Show context token usage")
        print_info("    /compact           Manually compact conversation history")
        print_info("    /soul              Show loaded SOUL.md")
        print_info("    /skills            List discovered skills")
        print_info("    /memory            Show memory stats")
        print_info("    /search <query>    Search long-term memory")
        print_info("    /prompt            Preview composed system prompt")
        print_info("    /bootstrap         Show loaded bootstrap files")
        print_info("    /reload_intelligence Reload bootstrap files and skills")
        print_info("    /heartbeat         Show heartbeat status")
        print_info("    /trigger           Force heartbeat now")
        print_info("    /cron              List cron jobs")
        print_info("    /cron-trigger <id> Trigger a cron job")
        print_info("    /cron-reload       Reload CRON.json")
        print_info("    /delivery          Show delivery stats")
        print_info("    /queue             Show pending deliveries")
        print_info("    /failed            Show failed deliveries")
        print_info("    /retry-failed      Retry failed deliveries")
        print_info("    /inbox             Show heartbeat/cron/delivery background output")
        print_info("    /inbox clear       Clear background inbox")
        print_info("    /resilience        Show LLM retry/circuit state")
        print_info("    /tool-mode [mode]  Show or set tool mode: safe/dev/trusted")
        print_info("    /lanes             Show named lane status")
        print_info("    /lane-queue        Show active/queued lane work")
        print_info("    /concurrency <lane> <N> Set lane concurrency")
        print_info("    /generation        Show lane generation counters")
        print_info("    /reset-lanes       Increment all lane generations")
        print_info("")
        print_info("  Gateway Commands:")
        print_info("    /bindings          List all route bindings")
        print_info("    /route <ch> <peer> Test route resolution")
        print_info("    /agents            List all agents")
        print_info("    /switch_agent <id> Force use specific agent")
        print_info("    /switch_agent off  Restore normal routing")
        print_info("    /gateway start     Start WebSocket gateway")
        print_info("    /gateway stop     Stop WebSocket gateway")
        print_info("    /gateway status   Show gateway status")
        print_info("    /sessions         List current sessions")
        print_info("")
        print_info("  Other Commands:")
        print_info("    /channels          List all channels")
        print_info("    /accounts          List configured accounts")
        print_info("    /help              Show this help")
        print_info("    quit / exit        Exit the REPL")
        return True, messages, force_agent_id, gw_server

    return False, messages, force_agent_id, gw_server
