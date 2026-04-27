from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .lanes import CommandQueue, LANE_MAIN
from .llm import LLMClient, create_llm_client
from .runtime import (
    AgentManager,
    Binding,
    BindingTable,
    build_session_key,
    normalize_agent_id,
)

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

class GatewayServer:
    def __init__(self, mgr: AgentManager, bindings: BindingTable,
                 llm_client: LLMClient | None = None,
                 command_queue: CommandQueue | None = None,
                 host: str = "localhost", port: int = 8765,
                 run_agent_func: Any | None = None,
                 run_async_func: Any | None = None,
                 resolve_route_func: Any | None = None) -> None:
        self._mgr = mgr
        self._bindings = bindings
        self._llm_client = llm_client or create_llm_client()
        self._command_queue = command_queue or CommandQueue()
        self._command_queue.get_or_create_lane(LANE_MAIN, max_concurrency=1)
        self._run_agent = run_agent_func
        self._run_async = run_async_func
        self._resolve_route = resolve_route_func
        self._host, self._port = host, port
        self._clients: set[Any] = set()
        self._start_time = time.monotonic()
        self._server: Any = None
        self._running = False

    async def start(self) -> None:
        try:
            import websockets
        except ImportError:
            raise RuntimeError("websockets not installed. pip install websockets")
        self._start_time = time.monotonic()
        self._server = await websockets.serve(self._handle, self._host, self._port)
        self._running = True
        print(f"{GREEN}Gateway started ws://{self._host}:{self._port}{RESET}")

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._running = False

    async def _handle(self, ws: Any, path: str = "") -> None:
        self._clients.add(ws)
        try:
            async for raw in ws:
                resp = await self._dispatch(raw)
                if resp:
                    await ws.send(json.dumps(resp))
        except Exception:
            pass
        finally:
            self._clients.discard(ws)

    def _typing_cb(self, agent_id: str, typing: bool) -> None:
        msg = json.dumps({"jsonrpc": "2.0", "method": "typing",
                          "params": {"agent_id": agent_id, "typing": typing}})
        for ws in list(self._clients):
            try:
                asyncio.ensure_future(ws.send(msg))
            except Exception:
                self._clients.discard(ws)

    async def _dispatch(self, raw: str) -> dict | None:
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            return {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}
        rid, method, params = req.get("id"), req.get("method", ""), req.get("params", {})
        methods = {
            "send": self._m_send, "bindings.set": self._m_bind_set,
            "bindings.list": self._m_bind_list, "sessions.list": self._m_sessions,
            "agents.list": self._m_agents, "status": self._m_status,
        }
        handler = methods.get(method)
        if not handler:
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Unknown: {method}"}, "id": rid}
        try:
            return {"jsonrpc": "2.0", "result": await handler(params), "id": rid}
        except Exception as exc:
            return {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(exc)}, "id": rid}

    async def _m_send(self, p: dict) -> dict:
        if self._run_agent is None:
            raise RuntimeError("run_agent_func is not configured")
        if self._run_async is None:
            raise RuntimeError("run_async_func is not configured")

        text = p.get("text", "")
        if not text:
            raise ValueError("text is required")
        ch = p.get("channel", "websocket")
        pid = p.get("peer_id", "ws-client")
        acc = p.get("account_id", "")
        gid = p.get("guild_id", "")
        if p.get("agent_id"):
            aid = normalize_agent_id(p["agent_id"])
            a = self._mgr.get_agent(aid)
            sk = build_session_key(aid, channel=ch, account_id=acc, peer_id=pid,
                                   dm_scope=a.dm_scope if a else "per-peer")
        else:
            if self._resolve_route is None:
                raise RuntimeError("resolve_route_func is not configured")
            aid, sk = self._resolve_route(
                self._bindings,
                self._mgr,
                channel=ch,
                peer_id=pid,
                account_id=acc,
                guild_id=gid,
            )
        future = self._command_queue.enqueue(
            LANE_MAIN,
            lambda: self._run_async(self._run_agent(
                self._mgr,
                aid,
                sk,
                text,
                on_typing=self._typing_cb,
                channel=ch,
                llm_client=self._llm_client,
            )),
        )
        reply = await asyncio.to_thread(future.result)
        return {"agent_id": aid, "session_key": sk, "reply": reply}

    async def _m_bind_set(self, p: dict) -> dict:
        b = Binding(agent_id=normalize_agent_id(p.get("agent_id", "")),
                    tier=int(p.get("tier", 5)), match_key=p.get("match_key", "default"),
                    match_value=p.get("match_value", "*"), priority=int(p.get("priority", 0)))
        self._bindings.add(b)
        return {"ok": True, "binding": b.display()}

    async def _m_bind_list(self, p: dict) -> list[dict]:
        return [{"agent_id": b.agent_id, "tier": b.tier, "match_key": b.match_key,
                 "match_value": b.match_value, "priority": b.priority}
                for b in self._bindings.list_all()]

    async def _m_sessions(self, p: dict) -> dict:
        return self._mgr.list_sessions(p.get("agent_id", ""))

    async def _m_agents(self, p: dict) -> list[dict]:
        return [{"id": a.id, "name": a.name, "model": a.effective_model,
                 "dm_scope": a.dm_scope, "personality": a.personality}
                for a in self._mgr.list_agents()]

    async def _m_status(self, p: dict) -> dict:
        return {"running": self._running,
                "uptime_seconds": round(time.monotonic() - self._start_time, 1),
                "connected_clients": len(self._clients),
                "agent_count": len(self._mgr.list_agents()),
                "binding_count": len(self._bindings.list_all())}
