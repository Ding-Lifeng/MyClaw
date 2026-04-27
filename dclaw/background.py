from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

try:
    from croniter import croniter
    HAS_CRONITER = True
except ImportError:
    croniter = None
    HAS_CRONITER = False

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

from .lanes import CommandQueue, LANE_CRON, LANE_HEARTBEAT
from .llm import LLMClient
from .runtime import AgentManager, DEFAULT_AGENT_ID, build_session_key, normalize_agent_id

# 工作目录
WORKDIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = WORKDIR / "workspace-main"
CRON_DIR = WORKSPACE_DIR / "cron"

class HeartbeatRunner:
    def __init__(
            self,
            workspace: Path,
            command_queue: CommandQueue,
            agent_mgr: AgentManager,
            llm_client: LLMClient,
            interval: float = 1800.0,
            active_hours: tuple[int, int] = (9, 22),
            agent_id: str = DEFAULT_AGENT_ID,
            run_agent_func: Any | None = None,
    ) -> None:
        self.workspace = workspace
        self.heartbeat_path = workspace / "HEARTBEAT.md"
        self.command_queue = command_queue
        self.agent_mgr = agent_mgr
        self.llm_client = llm_client
        self.interval = interval
        self.active_hours = active_hours
        self.agent_id = normalize_agent_id(agent_id)
        self.run_agent_func = run_agent_func
        run_on_startup = os.getenv("HEARTBEAT_RUN_ON_STARTUP", "").lower() in ("1", "true", "yes")
        self.last_run_at: float = 0.0 if run_on_startup else time.time()
        self.running = False
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._queue_lock = threading.Lock()
        self._output_queue: list[str] = []
        self._last_output = ""

    def _session_key(self) -> str:
        agent = self.agent_mgr.get_agent(self.agent_id)
        return build_session_key(
            self.agent_id,
            channel="heartbeat",
            account_id="system",
            peer_id="timer",
            dm_scope=agent.dm_scope if agent else "per-peer",
        )

    def should_run(self) -> tuple[bool, str]:
        if not self.heartbeat_path.exists():
            return False, "HEARTBEAT.md not found"
        instructions = self.heartbeat_path.read_text(encoding="utf-8").strip()
        if not instructions:
            return False, "HEARTBEAT.md is empty"
        now = time.time()
        elapsed = now - self.last_run_at
        if elapsed < self.interval:
            return False, f"interval not elapsed ({self.interval - elapsed:.0f}s remaining)"
        hour = datetime.now().hour
        start, end = self.active_hours
        in_hours = (start <= hour < end) if start <= end else not (end <= hour < start)
        if not in_hours:
            return False, f"outside active hours ({start}:00-{end}:00)"
        if self.running:
            return False, "already running"
        lane_stats = self.command_queue.get_or_create_lane(LANE_HEARTBEAT).stats()
        if lane_stats["active"] > 0:
            return False, "heartbeat lane busy"
        return True, "all checks passed"

    def _parse_response(self, response: str) -> str | None:
        stripped = response.strip()
        if not stripped:
            return None
        if "HEARTBEAT_OK" in stripped:
            stripped = stripped.replace("HEARTBEAT_OK", "").strip()
            return stripped or None
        return stripped

    def _build_prompt(self) -> str:
        instructions = self.heartbeat_path.read_text(encoding="utf-8").strip()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"{instructions}\n\n"
            "If there is nothing meaningful to report, reply with exactly HEARTBEAT_OK.\n"
            f"Current time: {now_str}"
        )

    def _queue_output(self, text: str) -> None:
        with self._queue_lock:
            self._output_queue.append(text)

    def _run_once(self) -> str | None:
        self.running = True
        try:
            if self.run_agent_func is None:
                raise RuntimeError("run_agent_func is not configured")
            response = run_async(self.run_agent_func(
                self.agent_mgr,
                self.agent_id,
                self._session_key(),
                self._build_prompt(),
                channel="heartbeat",
                llm_client=self.llm_client,
            ))
            meaningful = self._parse_response(response)
            if meaningful is None:
                return
            if meaningful == self._last_output:
                return None
            self._last_output = meaningful
            return meaningful
        except Exception as exc:
            return f"[heartbeat error: {exc}]"
        finally:
            self.running = False
            self.last_run_at = time.time()

    def _enqueue(self) -> concurrent.futures.Future:
        future = self.command_queue.enqueue(LANE_HEARTBEAT, self._run_once)

        def _on_done(done: concurrent.futures.Future) -> None:
            try:
                output = done.result()
                if output:
                    self._queue_output(output)
            except Exception as exc:
                self._queue_output(f"[heartbeat error: {exc}]")

        future.add_done_callback(_on_done)
        return future

    def _loop(self) -> None:
        while not self._stopped:
            try:
                ok, _ = self.should_run()
                if ok:
                    self._enqueue()
            except Exception:
                pass
            time.sleep(1.0)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="heartbeat")
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def drain_output(self) -> list[str]:
        with self._queue_lock:
            items = list(self._output_queue)
            self._output_queue.clear()
            return items

    def trigger(self) -> str:
        if not self.heartbeat_path.exists():
            return "HEARTBEAT.md not found"
        if not self.heartbeat_path.read_text(encoding="utf-8").strip():
            return "HEARTBEAT.md is empty"
        future = self._enqueue()
        try:
            output = future.result(timeout=float(os.getenv("HEARTBEAT_TRIGGER_TIMEOUT", "120")))
            if output is None:
                return "HEARTBEAT_OK or duplicate content (nothing queued)"
            return f"triggered, output queued ({len(output)} chars)"
        except concurrent.futures.TimeoutError:
            return "heartbeat queued; still running"
        except Exception as exc:
            return f"trigger failed: {exc}"

    def status(self) -> dict[str, Any]:
        now = time.time()
        elapsed = now - self.last_run_at if self.last_run_at > 0 else None
        next_in = max(0.0, self.interval - elapsed) if elapsed is not None else self.interval
        ok, reason = self.should_run()
        with self._queue_lock:
            qsize = len(self._output_queue)
        return {
            "enabled": self.heartbeat_path.exists(),
            "running": self.running,
            "should_run": ok,
            "reason": reason,
            "last_run": datetime.fromtimestamp(self.last_run_at).isoformat() if self.last_run_at > 0 else "never",
            "next_in": f"{round(next_in)}s",
            "interval": f"{self.interval}s",
            "active_hours": f"{self.active_hours[0]}:00-{self.active_hours[1]}:00",
            "queue_size": qsize,
            "agent_id": self.agent_id,
        }

CRON_AUTO_DISABLE_THRESHOLD = 5

@dataclass
class CronJob:
    id: str
    name: str
    enabled: bool
    schedule_kind: str
    schedule_config: dict[str, Any]
    payload: dict[str, Any]
    delete_after_run: bool = False
    consecutive_errors: int = 0
    last_run_at: float = 0.0
    next_run_at: float = 0.0


class CronService:
    def __init__(
            self,
            cron_file: Path,
            command_queue: CommandQueue,
            agent_mgr: AgentManager,
            llm_client: LLMClient,
            default_agent_id: str = DEFAULT_AGENT_ID,
            run_agent_func: Any | None = None,
    ) -> None:
        self.cron_file = cron_file
        self.command_queue = command_queue
        self.agent_mgr = agent_mgr
        self.llm_client = llm_client
        self.default_agent_id = normalize_agent_id(default_agent_id)
        self.run_agent_func = run_agent_func
        self.jobs: list[CronJob] = []
        self.running_job_id: str | None = None
        self._queue_lock = threading.Lock()
        self._output_queue: list[str] = []
        CRON_DIR.mkdir(parents=True, exist_ok=True)
        self._run_log = CRON_DIR / "cron-runs.jsonl"
        self.load_jobs()

    def load_jobs(self) -> None:
        self.jobs.clear()
        if not self.cron_file.exists():
            return
        try:
            raw = json.loads(self.cron_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self._queue_output(f"CRON.json load error: {exc}")
            return
        now = time.time()
        for job_data in raw.get("jobs", []):
            sched = job_data.get("schedule", {})
            kind = sched.get("kind", "")
            if kind not in ("at", "every", "cron"):
                continue
            job = CronJob(
                id=str(job_data.get("id", "")).strip(),
                name=str(job_data.get("name", "")).strip(),
                enabled=bool(job_data.get("enabled", True)),
                schedule_kind=kind,
                schedule_config=sched,
                payload=job_data.get("payload", {}),
                delete_after_run=bool(job_data.get("delete_after_run", False)),
            )
            if not job.id:
                continue
            if not job.name:
                job.name = job.id
            job.next_run_at = self._compute_next(job, now)
            self.jobs.append(job)

    def reload(self) -> str:
        self.load_jobs()
        return f"loaded {len(self.jobs)} cron job(s)"

    def _queue_output(self, text: str) -> None:
        with self._queue_lock:
            self._output_queue.append(text)

    def _parse_iso_timestamp(self, value: Any) -> float:
        if not value:
            raise ValueError("empty timestamp")
        return datetime.fromisoformat(str(value)).timestamp()

    def _cron_base_time(self, cfg: dict[str, Any], now: float) -> datetime:
        tz_name = cfg.get("tz")
        if tz_name and ZoneInfo is not None:
            try:
                return datetime.fromtimestamp(now, tz=ZoneInfo(str(tz_name)))
            except Exception:
                pass
        return datetime.fromtimestamp(now)

    @staticmethod
    def _parse_cron_field(field: str, min_value: int, max_value: int) -> set[int]:
        values: set[int] = set()
        for part in field.split(","):
            part = part.strip()
            if not part:
                continue
            if "/" in part:
                base, step_text = part.split("/", 1)
                step = int(step_text)
            else:
                base, step = part, 1
            if step <= 0:
                raise ValueError("cron step must be positive")
            if base == "*":
                start, end = min_value, max_value
            elif "-" in base:
                start_text, end_text = base.split("-", 1)
                start, end = int(start_text), int(end_text)
            else:
                start = end = int(base)
            if start < min_value or end > max_value or start > end:
                raise ValueError("cron field out of range")
            values.update(range(start, end + 1, step))
        return values

    @classmethod
    def _next_cron_fallback(cls, expr: str, base: datetime) -> float:
        fields = expr.split()
        if len(fields) != 5:
            raise ValueError("fallback cron parser supports 5 fields")
        minutes = cls._parse_cron_field(fields[0], 0, 59)
        hours = cls._parse_cron_field(fields[1], 0, 23)
        month_days = cls._parse_cron_field(fields[2], 1, 31)
        months = cls._parse_cron_field(fields[3], 1, 12)
        weekdays = cls._parse_cron_field(fields[4], 0, 7)
        if 7 in weekdays:
            weekdays.add(0)
            weekdays.discard(7)

        dom_restricted = fields[2].strip() != "*"
        dow_restricted = fields[4].strip() != "*"
        candidate = base.replace(second=0, microsecond=0) + timedelta(minutes=1)
        deadline = candidate + timedelta(days=366)
        while candidate <= deadline:
            cron_weekday = (candidate.weekday() + 1) % 7
            dom_match = candidate.day in month_days
            dow_match = cron_weekday in weekdays
            if dom_restricted and dow_restricted:
                day_match = dom_match or dow_match
            else:
                day_match = dom_match and dow_match
            if (
                    candidate.minute in minutes
                    and candidate.hour in hours
                    and candidate.month in months
                    and day_match
            ):
                return candidate.timestamp()
            candidate += timedelta(minutes=1)
        return 0.0

    def _compute_next(self, job: CronJob, now: float) -> float:
        cfg = job.schedule_config
        if job.schedule_kind == "at":
            try:
                ts = self._parse_iso_timestamp(cfg.get("at", ""))
                return ts if ts > now else 0.0
            except (ValueError, OSError, TypeError):
                return 0.0

        if job.schedule_kind == "every":
            try:
                every = float(cfg.get("every_seconds", 3600))
            except (TypeError, ValueError):
                return 0.0
            if every <= 0:
                return 0.0
            try:
                anchor = self._parse_iso_timestamp(cfg.get("anchor", ""))
            except (ValueError, OSError, TypeError):
                anchor = now
            if now < anchor:
                return anchor
            steps = int((now - anchor) / every) + 1
            return anchor + steps * every

        if job.schedule_kind == "cron":
            expr = str(cfg.get("expr", "")).strip()
            if not expr:
                return 0.0
            try:
                base = self._cron_base_time(cfg, now)
                if HAS_CRONITER and croniter is not None:
                    return croniter(expr, base).get_next(datetime).timestamp()
                return self._next_cron_fallback(expr, base)
            except Exception:
                return 0.0

        return 0.0

    def tick(self) -> None:
        now = time.time()
        remove_ids: list[str] = []
        for job in list(self.jobs):
            if not job.enabled or job.next_run_at <= 0 or now < job.next_run_at:
                continue
            ran = self._run_job(job, now)
            if ran and job.delete_after_run and job.schedule_kind == "at":
                remove_ids.append(job.id)
        if remove_ids:
            self.jobs = [job for job in self.jobs if job.id not in remove_ids]

    def _session_key(self, job: CronJob, agent_id: str) -> str:
        agent = self.agent_mgr.get_agent(agent_id)
        return build_session_key(
            agent_id,
            channel="cron",
            account_id="system",
            peer_id=job.id,
            dm_scope=agent.dm_scope if agent else "per-peer",
        )

    def _run_agent_payload(self, job: CronJob, payload: dict[str, Any]) -> str:
        message = str(payload.get("message", "")).strip()
        if not message:
            return "[empty message]"
        agent_id = normalize_agent_id(payload.get("agent_id") or self.default_agent_id)
        if not self.agent_mgr.get_agent(agent_id):
            raise ValueError(f"agent '{agent_id}' not found")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = (
            "Scheduled background task. Be concise and report only useful results.\n\n"
            f"Job: {job.name} ({job.id})\n"
            f"Current time: {now_str}\n\n"
            f"{message}"
        )
        if self.run_agent_func is None:
            raise RuntimeError("run_agent_func is not configured")
        return run_async(self.run_agent_func(
            self.agent_mgr,
            agent_id,
            self._session_key(job, agent_id),
            prompt,
            channel="cron",
            llm_client=self.llm_client,
        ))

    def _run_job(self, job: CronJob, now: float) -> bool:
        lane_stats = self.command_queue.get_or_create_lane(LANE_CRON).stats()
        if lane_stats["active"] > 0:
            return False

        job.next_run_at = self._compute_next(job, now)

        def _do_job() -> tuple[str, str, str]:
            self.running_job_id = job.id
            output, status, error = "", "ok", ""
            try:
                payload = job.payload
                kind = payload.get("kind", "")
                if kind == "agent_turn":
                    output = self._run_agent_payload(job, payload)
                    if output == "[empty message]":
                        status = "skipped"
                elif kind == "system_event":
                    output = str(payload.get("text", "")).strip()
                    if not output:
                        status = "skipped"
                else:
                    output = f"[unknown cron payload kind: {kind}]"
                    status = "error"
                    error = f"unknown payload kind: {kind}"
            except Exception as exc:
                status = "error"
                error = str(exc)
                output = f"[cron error: {exc}]"
            finally:
                self.running_job_id = None
            return output, status, error

        future = self.command_queue.enqueue(LANE_CRON, _do_job)

        def _on_done(done: concurrent.futures.Future, target: CronJob = job, run_at: float = now) -> None:
            try:
                output, status, error = done.result()
            except Exception as exc:
                output, status, error = f"[cron error: {exc}]", "error", str(exc)
            target.last_run_at = run_at
            if status == "error":
                target.consecutive_errors += 1
                if target.consecutive_errors >= CRON_AUTO_DISABLE_THRESHOLD:
                    target.enabled = False
                    self._queue_output(
                        f"Job '{target.name}' auto-disabled after {target.consecutive_errors} consecutive errors: {error}"
                    )
            else:
                target.consecutive_errors = 0
            self._append_log(target, run_at, status, output, error)
            if output and status != "skipped":
                self._queue_output(f"[{target.name}] {output}")

        future.add_done_callback(_on_done)
        return True

    def _append_log(self, job: CronJob, run_at: float, status: str, output: str, error: str = "") -> None:
        entry = {
            "job_id": job.id,
            "job_name": job.name,
            "run_at": datetime.fromtimestamp(run_at, tz=timezone.utc).isoformat(),
            "status": status,
            "output_preview": output[:200],
        }
        if error:
            entry["error"] = error
        try:
            with self._run_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def trigger_job(self, job_id: str) -> str:
        target = job_id.strip()
        for job in self.jobs:
            if job.id == target:
                ran = self._run_job(job, time.time())
                if not ran:
                    return "main lane occupied, cannot trigger"
                return f"'{job.name}' triggered (errors={job.consecutive_errors})"
        return f"Job '{target}' not found"

    def drain_output(self) -> list[str]:
        with self._queue_lock:
            items = list(self._output_queue)
            self._output_queue.clear()
            return items

    def list_jobs(self) -> list[dict[str, Any]]:
        now = time.time()
        result: list[dict[str, Any]] = []
        for job in self.jobs:
            next_in = max(0.0, job.next_run_at - now) if job.next_run_at > 0 else None
            result.append({
                "id": job.id,
                "name": job.name,
                "enabled": job.enabled,
                "kind": job.schedule_kind,
                "errors": job.consecutive_errors,
                "last_run": datetime.fromtimestamp(job.last_run_at).isoformat() if job.last_run_at > 0 else "never",
                "next_run": datetime.fromtimestamp(job.next_run_at).isoformat() if job.next_run_at > 0 else "n/a",
                "next_in": round(next_in) if next_in is not None else None,
            })
        return result

    def status(self) -> dict[str, Any]:
        return {
            "cron_file": str(self.cron_file),
            "jobs": len(self.jobs),
            "running_job_id": self.running_job_id or "",
            "croniter": HAS_CRONITER,
            "log": str(self._run_log),
        }

_event_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None

def get_event_loop() -> asyncio.AbstractEventLoop:
    global _event_loop, _loop_thread
    if _event_loop is not None and _event_loop.is_running():
        return _event_loop
    _event_loop = asyncio.new_event_loop()
    def _run():
        asyncio.set_event_loop(_event_loop)
        _event_loop.run_forever()
    _loop_thread = threading.Thread(target=_run, daemon=True)
    _loop_thread.start()
    return _event_loop

def run_async(coro: Any) -> Any:
    loop = get_event_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result()
