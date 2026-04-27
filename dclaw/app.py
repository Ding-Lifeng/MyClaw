import sys
import queue, threading
from pathlib import Path
import dashscope
from dclaw.config import CONFIG, MODEL_ID
from dclaw.channel_setup import register_configured_channels
from dclaw.intelligence_runtime import IntelligenceRuntime
from dclaw.runtime_context import RuntimeContext

# ---------------------------------------------------------------------------
# API配置
# ---------------------------------------------------------------------------

# 加载环境变量
dashscope.base_http_api_url = CONFIG.llm.dashscope_base_url

# 工作目录
WORKDIR = CONFIG.project_root
WORKSPACE_DIR = CONFIG.workspace_dir
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
CRON_DIR = CONFIG.cron_dir
DELIVERY_DIR = CONFIG.delivery_dir

# Agents 目录 -- 多Agent
AGENTS_DIR = CONFIG.agents_dir

# 状态目录 -- 存储Agent运行过程文件
STATE_DIR = CONFIG.state_dir
STATE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ANSI 颜色配置-丰富终端显示效果
# ---------------------------------------------------------------------------
from dclaw.terminal import (
    DIM,
    RED,
    RESET,
    BackgroundInbox,
    colored_prompt,
    print_assistant,
    print_channel,
    print_delivery,
    print_info,
    print_session,
    print_tool,
    print_warn,
)

# ---------------------------------------------------------------------------
# Bootstrap 文件加载
# ---------------------------------------------------------------------------

from dclaw.runtime import (
    DEFAULT_AGENT_ID,
    AgentManager,
    BindingTable,
    SessionStore,
    build_session_key,
)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

# 统一输入数据结构
from dclaw.channels import (
    CLIChannel,
    ChannelManager,
    DefaultChannel,
    InboundMessage,
)

# ---------------------------------------------------------------------------
# Delivery - 持久化投递队列
# ---------------------------------------------------------------------------

from dclaw.delivery import (
    DeliveryQueue,
    DeliveryRunner,
    normalize_delivery_channel,
)
from dclaw.workspace import WorkspacePolicy

WORKSPACE_POLICY = WorkspacePolicy(project_root=WORKDIR, workspace_root=WORKSPACE_DIR)

def enqueue_delivery(delivery_queue: DeliveryQueue | None, channel: str, to: str, text: str) -> bool:
    if delivery_queue is None:
        return False
    ids = delivery_queue.enqueue(channel, to, text)
    if ids:
        print_delivery(f"queued {len(ids)} chunk(s) for {normalize_delivery_channel(channel)}:{to}")
    return bool(ids)

def safe_path(raw: str) -> Path:
    return WORKSPACE_POLICY.resolve_workspace_path(raw)

# ---------------------------------------------------------------------------
# LLM Adapter
# ---------------------------------------------------------------------------

from dclaw.llm import (
    create_llm_client,
)

# ---------------------------------------------------------------------------
# 处理会话消息-防止上下文溢出
# ---------------------------------------------------------------------------

from dclaw.context import ContextGuard


# 截断过长文本 TODO:统一truncate工具
from dclaw.tools import (
    TOOL_HANDLERS,
    configure_tools,
    get_tool_policy,
)

# ---------------------------------------------------------------------------
# 共享事件循环 (持久化后台线程)
# ---------------------------------------------------------------------------

INTELLIGENCE = IntelligenceRuntime(WORKSPACE_DIR)

def auto_recall(user_message: str, top_k: int = 3) -> str:
    return INTELLIGENCE.auto_recall(user_message, top_k=top_k)

def compose_runtime_system_prompt(
        agent_id: str,
        channel: str,
        memory_context: str = "",
        mode: str = "full",
) -> str:
    return INTELLIGENCE.compose_system_prompt(
        agent_id=agent_id,
        channel=channel,
        memory_context=memory_context,
        mode=mode,
    )

from dclaw.lanes import (
    CommandQueue,
    LANE_CRON,
    LANE_DELIVERY,
    LANE_HEARTBEAT,
    LANE_MAIN,
)

from dclaw.background import (
    CronService,
    HeartbeatRunner,
    run_async,
)

# ---------------------------------------------------------------------------
# 路由解析
# ---------------------------------------------------------------------------

from dclaw.engine import (
    EngineServices,
    resolve_route,
    run_agent,
)

# ---------------------------------------------------------------------------
# Gateway 服务器 (WebSocket, JSON-RPC 2.0)
# ---------------------------------------------------------------------------

from dclaw.gateway import GatewayServer

# ---------------------------------------------------------------------------
# 默认 Agent 初始化
# ---------------------------------------------------------------------------

from dclaw.runtime import setup_default_agent

# ---------------------------------------------------------------------------
# Read-Eval-Print Loop
# ---------------------------------------------------------------------------

from dclaw.repl import ReplServices, handle_repl_command

ENGINE_SERVICES: EngineServices | None = None
REPL_SERVICES: ReplServices | None = None

def approve_tool_request(tool_name: str, detail: str) -> bool:
    print_warn(f"Tool '{tool_name}' requires approval: {detail}")
    answer = input("Allow once? [y/N] ").strip().lower()
    return answer in ("y", "yes")

def configure_modules() -> None:
    global ENGINE_SERVICES, REPL_SERVICES
    configure_tools(
        print_tool_func=print_tool,
        workspace_policy=WORKSPACE_POLICY,
        mode="dev",
        approval_callback=approve_tool_request,
    )
    ENGINE_SERVICES = EngineServices(
        auto_recall=auto_recall,
        compose_runtime_system_prompt=compose_runtime_system_prompt,
        print_info=print_info,
        print_assistant=print_assistant,
        enqueue_delivery=enqueue_delivery,
    )
    REPL_SERVICES = ReplServices(
        auto_recall=auto_recall,
        refresh_intelligence=INTELLIGENCE.refresh,
        get_bootstrap_data=lambda: INTELLIGENCE.bootstrap_data,
        get_skills_block=lambda: INTELLIGENCE.skills_block,
        get_skills_manager=lambda: INTELLIGENCE.skills_manager,
    )

# ---------------------------------------------------------------------------
# Agent 交互回合
# ---------------------------------------------------------------------------

from dclaw.engine import run_agent_turn

# ---------------------------------------------------------------------------
# 核心: Agent 循环
# ---------------------------------------------------------------------------

def agent_loop() -> None:
    configure_modules()

    try:
        llm_client = create_llm_client()
    except Exception as e:
        print(f"{RED}Failed to create LLM client: {e}{RESET}")
        sys.exit(1)

    mgr = ChannelManager(notify=print_channel)
    bindings = BindingTable()
    store = SessionStore()
    agent_mgr = AgentManager(session_store=store)

    INTELLIGENCE.refresh()
    setup_default_agent(agent_mgr, bindings)

    cli = CLIChannel(prompt_func=colored_prompt, send_func=print_assistant)
    mgr.register(cli)
    mgr.register(DefaultChannel(send_func=print_assistant))

    before_channels = set(mgr.list_channels())
    register_configured_channels(CONFIG, mgr)
    for channel_name in set(mgr.list_channels()) - before_channels:
        if channel_name == "feishu":
            print_channel("[+] Feishu channel registered (requires webhook server)")
        elif channel_name == "wechat":
            print_channel("[+] Wechat channel registered (webhook delivery)")

    guard = ContextGuard()
    command_queue = CommandQueue()
    command_queue.get_or_create_lane(LANE_MAIN, max_concurrency=1)
    command_queue.get_or_create_lane(LANE_HEARTBEAT, max_concurrency=1)
    command_queue.get_or_create_lane(LANE_CRON, max_concurrency=1)
    command_queue.get_or_create_lane(LANE_DELIVERY, max_concurrency=1)

    engine_services = ENGINE_SERVICES or EngineServices(
        auto_recall=auto_recall,
        compose_runtime_system_prompt=compose_runtime_system_prompt,
        print_info=print_info,
        print_assistant=print_assistant,
        enqueue_delivery=enqueue_delivery,
    )
    repl_services = REPL_SERVICES or ReplServices(
        auto_recall=auto_recall,
        refresh_intelligence=INTELLIGENCE.refresh,
        get_bootstrap_data=lambda: INTELLIGENCE.bootstrap_data,
        get_skills_block=lambda: INTELLIGENCE.skills_block,
        get_skills_manager=lambda: INTELLIGENCE.skills_manager,
    )

    def run_agent_with_services(*args, **kwargs):
        kwargs.setdefault("services", engine_services)
        return run_agent(*args, **kwargs)

    def resolve_route_with_services(*args, **kwargs):
        kwargs.setdefault("services", engine_services)
        return resolve_route(*args, **kwargs)

    repl_services.run_agent = run_agent_with_services
    repl_services.resolve_route = resolve_route_with_services

    inbox = BackgroundInbox()
    delivery_queue = DeliveryQueue(DELIVERY_DIR)
    delivery_runner = DeliveryRunner(
        delivery_queue,
        mgr,
        command_queue=command_queue,
        inbox=inbox,
        notify=print_delivery,
    )
    delivery_runner.start()

    conversations: dict[str, list[dict]] = {}

    cli_agent = agent_mgr.get_agent(DEFAULT_AGENT_ID)
    cli_sk = build_session_key(
        DEFAULT_AGENT_ID,
        channel="cli",
        account_id="cli-local",
        peer_id="cli-user",
        dm_scope=cli_agent.dm_scope if cli_agent else "per-peer",
    )
    cli_history = store.load_session(cli_sk)
    if cli_history:
        conversations[cli_sk] = cli_history
        print_session(f"Resumed session: {cli_sk} ({len(cli_history)} messages)")
    else:
        store.ensure_session(cli_sk, label="cli")
        conversations[cli_sk] = []
        print_session(f"Created initial session: {cli_sk}")

    msg_queue: queue.Queue[InboundMessage | None] = queue.Queue()
    stop_event = threading.Event()

    force_agent_id: str | None = None
    gw_server: GatewayServer | None = None
    runtime_context: RuntimeContext | None = None
    heartbeat = HeartbeatRunner(
        workspace=WORKSPACE_DIR,
        command_queue=command_queue,
        agent_mgr=agent_mgr,
        llm_client=llm_client,
        interval=CONFIG.runtime.heartbeat_interval,
        active_hours=(
            CONFIG.runtime.heartbeat_active_start,
            CONFIG.runtime.heartbeat_active_end,
        ),
        agent_id=DEFAULT_AGENT_ID,
        run_agent_func=run_agent_with_services,
    )
    heartbeat.start()
    cron_svc = CronService(
        WORKSPACE_DIR / "CRON.json",
        command_queue=command_queue,
        agent_mgr=agent_mgr,
        llm_client=llm_client,
        default_agent_id=DEFAULT_AGENT_ID,
        run_agent_func=run_agent_with_services,
    )
    cron_stop = threading.Event()

    def cron_loop() -> None:
        while not cron_stop.is_set():
            try:
                cron_svc.tick()
            except Exception:
                pass
            cron_stop.wait(timeout=1.0)

    def set_force_agent(aid: str | None) -> None:
        nonlocal force_agent_id
        force_agent_id = aid
        if runtime_context is not None:
            runtime_context.force_agent_id = aid

    def set_gw_server(gw: GatewayServer | None) -> None:
        nonlocal gw_server
        gw_server = gw
        if runtime_context is not None:
            runtime_context.gw_server = gw

    def cli_reader():
        while not stop_event.is_set():
            msg = cli.receive()
            if msg is None:
                continue
            msg_queue.put(msg)

    def collect_background_outputs() -> None:
        for item in heartbeat.drain_output():
            inbox.add("heartbeat", item)
        for item in cron_svc.drain_output():
            inbox.add("cron", item)

    runtime_context = RuntimeContext(
        mgr=mgr,
        bindings=bindings,
        store=store,
        agent_mgr=agent_mgr,
        guard=guard,
        llm_client=llm_client,
        command_queue=command_queue,
        delivery_queue=delivery_queue,
        delivery_runner=delivery_runner,
        inbox=inbox,
        heartbeat=heartbeat,
        cron_svc=cron_svc,
        active_session_key=cli_sk,
        bootstrap_data=INTELLIGENCE.bootstrap_data,
        skills_mgr=INTELLIGENCE.skills_manager,
        skills_block=INTELLIGENCE.skills_block,
        force_agent_id=force_agent_id,
        gw_server=gw_server,
        set_force_agent=set_force_agent,
        set_gw_server=set_gw_server,
    )

    print_info("=" * 60)
    print_info(f" Provider: {type(llm_client).__name__}")
    print_info(f" Model: {MODEL_ID}")
    print_info(f" Session: {store.current_session_id}")
    print_info(f" Channels: {', '.join(mgr.list_channels())}")
    print_info(f" Agents: {', '.join(a.id for a in agent_mgr.list_agents())}")
    print_info(f" Bindings: {len(bindings.list_all())}")
    print_info(f" Intelligence workspace: {WORKSPACE_DIR}")
    print_info(f" Bootstrap files: {len(INTELLIGENCE.bootstrap_data)}")
    print_info(f" Skills: {len(INTELLIGENCE.skills_manager.skills)}")
    print_info(f" Heartbeat: {'on' if heartbeat.heartbeat_path.exists() else 'off'} ({heartbeat.interval}s)")
    print_info(f" Cron jobs: {len(cron_svc.jobs)}")
    print_info(f" Delivery queue: {delivery_queue.queue_dir}")
    print_info(f" Workdir: {WORKDIR}")
    print_info(f" Tool mode: {get_tool_policy().mode.value}")
    print_info(f" Tools: {', '.join(TOOL_HANDLERS.keys())}")
    print_info("  输入 /help 获取指令提示, 输入 'quit' 或 'exit' 退出.")
    print_info("=" * 60)
    print()

    cli_thread = threading.Thread(target=cli_reader, daemon=True)
    cli_thread.start()
    cron_thread = threading.Thread(target=cron_loop, daemon=True, name="cron-tick")
    cron_thread.start()

    last_inbox_notice_count = 0

    try:
        while not stop_event.is_set():
            active_cli_sk = runtime_context.active_session_key if runtime_context is not None else cli_sk
            collect_background_outputs()
            try:
                msg = msg_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if msg is None:
                break

            collect_background_outputs()
            if msg.channel == "cli":
                inbox_count = inbox.count()
                if inbox_count and inbox_count != last_inbox_notice_count and not msg.text.startswith("/inbox"):
                    print_info(f"  Background inbox: {inbox_count} unread item(s). Use /inbox to view.")
                    last_inbox_notice_count = inbox_count

            if msg.channel == "cli":
                if msg.text.lower() in ("quit", "exit"):
                    stop_event.set()
                    break

                if msg.text.startswith("/"):
                    active_cli_sk = runtime_context.active_session_key if runtime_context is not None else cli_sk
                    current_messages = conversations.get(active_cli_sk, [])
                    runtime_context.force_agent_id = force_agent_id
                    runtime_context.gw_server = gw_server
                    runtime_context.bootstrap_data = INTELLIGENCE.bootstrap_data
                    runtime_context.skills_mgr = INTELLIGENCE.skills_manager
                    runtime_context.skills_block = INTELLIGENCE.skills_block
                    handled, new_messages, new_force, new_gw = handle_repl_command(
                        msg.text, store, guard, current_messages, mgr,
                        llm_client, MODEL_ID,
                        runtime_context=runtime_context,
                        services=repl_services,
                    )
                    active_cli_sk = runtime_context.active_session_key if runtime_context is not None else active_cli_sk
                    conversations[active_cli_sk] = new_messages
                    if new_force is not None:
                        force_agent_id = new_force
                    if new_gw is not None:
                        gw_server = new_gw
                    if handled:
                        cli.allow_input()
                        continue

            if msg.channel == "cli":
                if force_agent_id:
                    aid = force_agent_id
                    a = agent_mgr.get_agent(aid)
                    if a:
                        sk = build_session_key(aid, channel=msg.channel, account_id=msg.account_id,
                                               peer_id=msg.peer_id, dm_scope=a.dm_scope)
                        future = command_queue.enqueue(LANE_MAIN, lambda: run_async(run_agent_with_services(
                                agent_mgr, aid, sk, msg.text,
                                channel=msg.channel,
                                llm_client=llm_client,
                            )))
                        result = future.result()
                        enqueue_delivery(delivery_queue, "default", msg.peer_id, result)
                        delivery_runner.flush()
                    else:
                        print_warn(f"Agent '{aid}' not found")
                else:
                    active_cli_sk = runtime_context.active_session_key if runtime_context is not None else cli_sk
                    cli_agent = agent_mgr.get_agent(DEFAULT_AGENT_ID)
                    memory_context = auto_recall(msg.text)
                    system_prompt = compose_runtime_system_prompt(
                        agent_id=DEFAULT_AGENT_ID,
                        channel=msg.channel,
                        memory_context=memory_context,
                    )
                    future = command_queue.enqueue(LANE_MAIN, lambda: run_agent_turn(
                            msg,
                            conversations,
                            mgr,
                            llm_client,
                            store=store,
                            model_id=cli_agent.effective_model if cli_agent else MODEL_ID,
                            system_prompt=system_prompt,
                            session_key=active_cli_sk,
                            delivery_queue=delivery_queue,
                            services=engine_services,
                        ))
                    future.result()
                    delivery_runner.flush()
                cli.allow_input()
            else:
                if force_agent_id:
                    aid = force_agent_id
                    a = agent_mgr.get_agent(aid)
                    if a:
                        sk = build_session_key(aid, channel=msg.channel, account_id=msg.account_id,
                                               peer_id=msg.peer_id, dm_scope=a.dm_scope)
                        future = command_queue.enqueue(LANE_MAIN, lambda: run_async(run_agent_with_services(
                                agent_mgr, aid, sk, msg.text,
                                channel=msg.channel,
                                llm_client=llm_client,
                            )))
                        result = future.result()
                        enqueue_delivery(delivery_queue, msg.channel, msg.peer_id, result)
                        delivery_runner.flush()
                    else:
                        print_warn(f"Agent '{aid}' not found")
                else:
                    aid, sk = resolve_route_with_services(bindings, agent_mgr, channel=msg.channel,
                                                          account_id=msg.account_id, peer_id=msg.peer_id)
                    future = command_queue.enqueue(LANE_MAIN, lambda: run_async(run_agent_with_services(
                            agent_mgr, aid, sk, msg.text,
                            channel=msg.channel,
                            llm_client=llm_client,
                        )))
                    result = future.result()
                    enqueue_delivery(delivery_queue, msg.channel, msg.peer_id, result)
                    delivery_runner.flush()

    except KeyboardInterrupt:
        print()
    finally:
        stop_event.set()
        cron_stop.set()
        cli_thread.join(timeout=2.0)
        cron_thread.join(timeout=2.0)
        heartbeat.stop()
        delivery_runner.stop()
        if gw_server:
            run_async(gw_server.stop())
        mgr.close_all()
        print(f"{DIM}再见.{RESET}")

# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    agent_loop()

if __name__ == "__main__":
    main()
