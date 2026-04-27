from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from .channels import ChannelManager, InboundMessage
from .config import MODEL_ID
from .context import ContextGuard
from .delivery import DeliveryQueue
from .llm import AnthropicClient, LLMClient, create_llm_client
from .runtime import AgentManager, BindingTable, DEFAULT_AGENT_ID, SessionStore, build_session_key
from .tools import TOOLS, process_tool_call

DIM = "\033[2m"
YELLOW = "\033[33m"
RESET = "\033[0m"
SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools.\n"
    "You can also connect to multiple messaging channels.\n"
    "Use tools to help the user with file and time queries.\n"
    "Be concise. If a session has prior context, use it."
)
MAX_AGENT_TOOL_ITERATIONS = 20
MAX_REPEATED_TOOL_CALLS = 2

def _default_auto_recall(user_message: str, top_k: int = 3) -> str:
    return ""

def _default_compose_runtime_system_prompt(
        agent_id: str,
        channel: str,
        memory_context: str = "",
        mode: str = "full",
) -> str:
    return "You are a helpful AI assistant."

_auto_recall = _default_auto_recall
_compose_runtime_system_prompt = _default_compose_runtime_system_prompt
_print_info = print
_print_assistant = print
_enqueue_delivery = lambda delivery_queue, channel, to, text: False


@dataclass
class EngineServices:
    auto_recall: Any = _default_auto_recall
    compose_runtime_system_prompt: Any = _default_compose_runtime_system_prompt
    print_info: Any = print
    print_assistant: Any = print
    enqueue_delivery: Any = lambda delivery_queue, channel, to, text: False


DEFAULT_ENGINE_SERVICES = EngineServices()


def configure_engine(
        auto_recall_func=None,
        compose_runtime_system_prompt_func=None,
        print_info_func=None,
        print_assistant_func=None,
        enqueue_delivery_func=None,
) -> None:
    global _auto_recall, _compose_runtime_system_prompt, _print_info, _print_assistant, _enqueue_delivery
    if auto_recall_func is not None:
        _auto_recall = auto_recall_func
    if compose_runtime_system_prompt_func is not None:
        _compose_runtime_system_prompt = compose_runtime_system_prompt_func
    if print_info_func is not None:
        _print_info = print_info_func
    if print_assistant_func is not None:
        _print_assistant = print_assistant_func
    if enqueue_delivery_func is not None:
        _enqueue_delivery = enqueue_delivery_func

def get_engine_services() -> EngineServices:
    return EngineServices(
        auto_recall=_auto_recall,
        compose_runtime_system_prompt=_compose_runtime_system_prompt,
        print_info=_print_info,
        print_assistant=_print_assistant,
        enqueue_delivery=_enqueue_delivery,
    )

def _tool_signature(tool_name: str, tool_args: dict[str, Any]) -> str:
    try:
        args = json.dumps(tool_args, sort_keys=True, ensure_ascii=False)
    except TypeError:
        args = str(tool_args)
    return f"{tool_name}:{args}"

def _process_tool_call_guarded(
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_counts: dict[str, int],
) -> str:
    signature = _tool_signature(tool_name, tool_args)
    tool_call_counts[signature] = tool_call_counts.get(signature, 0) + 1
    if tool_call_counts[signature] > MAX_REPEATED_TOOL_CALLS:
        return (
            "Error: Repeated identical tool call skipped. "
            "Use the previous tool result or explain that the requested information could not be verified."
        )
    return process_tool_call(tool_name, tool_args)

def resolve_route(bindings: BindingTable, mgr: AgentManager,
                  channel: str, peer_id: str,
                  account_id: str = "", guild_id: str = "",
                  services: EngineServices | None = None) -> tuple[str, str]:
    svc = services or get_engine_services()
    agent_id, matched = bindings.resolve(
        channel=channel, account_id=account_id,
        guild_id=guild_id, peer_id=peer_id,
    )
    if not agent_id:
        agent_id = DEFAULT_AGENT_ID
        svc.print_info(f"{DIM}[route] No binding matched, default: {agent_id}{RESET}")
    elif matched:
        svc.print_info(f"{DIM}[route] Matched: {matched.display()}{RESET}")
    agent = mgr.get_agent(agent_id)
    dm_scope = agent.dm_scope if agent else "per-peer"
    sk = build_session_key(agent_id, channel=channel, account_id=account_id,
                           peer_id=peer_id, dm_scope=dm_scope)
    return agent_id, sk

# ---------------------------------------------------------------------------
# Agent 运行器 - 限制并发请求数量
# ---------------------------------------------------------------------------

_agent_semaphore: asyncio.Semaphore | None = None

async def run_agent(mgr: AgentManager, agent_id: str, session_key: str,
                    user_text: str, on_typing: Any = None,
                    channel: str = "unknown",
                    llm_client: LLMClient | None = None,
                    services: EngineServices | None = None) -> str:
    global _agent_semaphore
    svc = services or get_engine_services()
    if _agent_semaphore is None:
        _agent_semaphore = asyncio.Semaphore(4)
    if llm_client is None:
        llm_client = create_llm_client()
    agent = mgr.get_agent(agent_id)
    if not agent:
        return f"Error: agent '{agent_id}' not found"
    messages = mgr.get_session(session_key)
    messages.append({"role": "user", "content": user_text})
    memory_context = svc.auto_recall(user_text)
    system_prompt = svc.compose_runtime_system_prompt(
        agent_id=agent.id,
        channel=channel,
        memory_context=memory_context,
    )
    async with _agent_semaphore:
        if on_typing:
            on_typing(agent_id, True)
        try:
            return await _agent_loop(llm_client, agent.effective_model, system_prompt, messages)
        finally:
            mgr.save_session(session_key)
            if on_typing:
                on_typing(agent_id, False)

async def _agent_loop(client: LLMClient, model: str, system: str, messages: list[dict]) -> str:
    tool_call_counts: dict[str, int] = {}
    for _ in range(MAX_AGENT_TOOL_ITERATIONS):
        try:
            response = await asyncio.to_thread(
                client.chat,
                model=model,
                system=system,
                messages=messages,
                max_tokens=4096,
                tools=TOOLS,
            )
        except Exception as exc:
            while messages and messages[-1]["role"] != "user":
                messages.pop()
            if messages:
                messages.pop()
            return f"API Error: {exc}"

        finish_reason = response["finish_reason"]
        assistant_text = response["text"]
        tool_calls = response["tool_calls"]

        assistant_msg = {"role": "assistant", "content": assistant_text}
        if tool_calls:
            if isinstance(client, AnthropicClient):
                content_blocks = []
                if assistant_text:
                    content_blocks.append({"type": "text", "text": assistant_text})
                for tc in tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"].get("arguments", "{}")),
                    })
                assistant_msg["content"] = content_blocks
            else:
                assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if finish_reason == "stop":
            return assistant_text or "[no text]"

        if finish_reason == "tool_calls":
            if isinstance(client, AnthropicClient):
                tool_results = []
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"].get("arguments", "{}"))
                    tool_call_id = tc.get("id")
                    result = _process_tool_call_guarded(tool_name, tool_args, tool_call_counts)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})
            else:
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"].get("arguments", "{}"))
                    tool_call_id = tc.get("id")
                    result = _process_tool_call_guarded(tool_name, tool_args, tool_call_counts)
                    messages.append({"role": "tool", "content": result, "tool_call_id": tool_call_id})
            continue

        return assistant_text or f"[finish_reason={finish_reason}]"
    return "[max tool iterations reached]"

def run_agent_turn(
        inbound: InboundMessage,
        conversations: dict[str, list[dict]],
        mgr: ChannelManager,
        client: LLMClient,
        store: SessionStore | None = None,
        model_id: str = MODEL_ID,
        system_prompt: str = SYSTEM_PROMPT,
        session_key: str | None = None,
        delivery_queue: DeliveryQueue | None = None,
        services: EngineServices | None = None,
) -> None:
    svc = services or get_engine_services()
    sk = session_key or build_session_key(
        DEFAULT_AGENT_ID,
        channel=inbound.channel,
        account_id=inbound.account_id,
        peer_id=inbound.peer_id,
    )
    if sk not in conversations:
        conversations[sk] = []
    messages = conversations[sk]

    should_persisit = (store is not None and inbound.channel == "cli")
    if should_persisit:
        store.ensure_session(sk, label=inbound.channel)

    # --- 添加聊天记录到历史 ---
    user_message = {
        "role": "user",
        "content": inbound.text,
    }
    messages.append(user_message)
    if should_persisit:
        user_record = user_message.copy()
        store.append_transcript(sk, user_record)

    guard = ContextGuard()

    tool_call_counts: dict[str, int] = {}
    for _ in range(MAX_AGENT_TOOL_ITERATIONS):
        try:
            response = guard.guard_api_call(
                client=client,
                model=model_id,
                max_tokens=8096,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as exc:
            svc.print_info(f"\n{YELLOW}API Error: {exc}{RESET}\n")
            while messages and messages[-1]["role"] != "user":
                messages.pop()
            if messages:
                messages.pop()
            break

        finish_reason = response["finish_reason"]
        assistant_content = response["text"]
        tool_calls = response["tool_calls"]

        assistant_msg = {
            "role": "assistant",
            "content": assistant_content
        }

        if tool_calls:
            if isinstance(client, AnthropicClient):
                content_blocks = []
                if assistant_content:
                    content_blocks.append({"type": "text", "text": assistant_content})
                for tc in tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]),
                    })
                assistant_msg["content"] = content_blocks
            else:
                assistant_msg["tool_calls"] = tool_calls

        messages.append(assistant_msg)
        if should_persisit:
            store.append_transcript(sk, assistant_msg.copy())

        # --- 调用终止条件stop_reason ---
        if finish_reason == "stop":
            if assistant_content:
                if not svc.enqueue_delivery(delivery_queue, inbound.channel, inbound.peer_id, assistant_content):
                    ch = mgr.get(inbound.channel)
                    if ch:
                        ch.send(inbound.peer_id, assistant_content)
                    else:
                        svc.print_assistant(assistant_content)
            break

        elif finish_reason == "tool_calls":
            if isinstance(client, AnthropicClient):
                tool_results = []
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"]["arguments"])
                    tool_call_id = tc["id"]

                    result = _process_tool_call_guarded(tool_name, tool_args, tool_call_counts)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": result
                    })

                tool_msg = {
                    "role": "user",
                    "content": tool_results
                }
                messages.append(tool_msg)
                if should_persisit:
                    store.append_transcript(sk, tool_msg.copy())
            else:
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"]["arguments"])
                    tool_call_id = tc["id"]

                    result = _process_tool_call_guarded(tool_name, tool_args, tool_call_counts)
                    tool_msg = {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call_id
                    }
                    messages.append(tool_msg)

                    if should_persisit:
                        store.append_transcript(sk, tool_msg.copy())
            continue

        else:
            svc.print_info(f"[finish_reason]={finish_reason}")
            if assistant_content:
                if not svc.enqueue_delivery(delivery_queue, inbound.channel, inbound.peer_id, assistant_content):
                    ch = mgr.get(inbound.channel)
                    if ch:
                        ch.send(inbound.peer_id, assistant_content)
                    else:
                        svc.print_assistant(assistant_content)
            break
    else:
        message = "[max tool iterations reached]"
        if not svc.enqueue_delivery(delivery_queue, inbound.channel, inbound.peer_id, message):
            ch = mgr.get(inbound.channel)
            if ch:
                ch.send(inbound.peer_id, message)
            else:
                svc.print_assistant(message)
