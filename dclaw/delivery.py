from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .lanes import CommandQueue, LANE_DELIVERY


DELIVERY_BACKOFF_MS = [5_000, 25_000, 120_000, 600_000]
DELIVERY_MAX_RETRIES = 5
DELIVERY_CHANNEL_LIMITS = {"default": 4096, "feishu": 4096, "wechat": 2048}


@dataclass
class QueuedDelivery:
    id: str
    channel: str
    to: str
    text: str
    retry_count: int = 0
    last_error: str = ""
    enqueued_at: float = field(default_factory=time.time)
    next_retry_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "channel": self.channel,
            "to": self.to,
            "text": self.text,
            "retry_count": self.retry_count,
            "last_error": self.last_error,
            "enqueued_at": self.enqueued_at,
            "next_retry_at": self.next_retry_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "QueuedDelivery":
        return QueuedDelivery(
            id=str(data["id"]),
            channel=str(data["channel"]).lower(),
            to=str(data["to"]),
            text=str(data["text"]),
            retry_count=int(data.get("retry_count", 0)),
            last_error=str(data.get("last_error", "")),
            enqueued_at=float(data.get("enqueued_at", 0.0)),
            next_retry_at=float(data.get("next_retry_at", 0.0)),
        )


def normalize_delivery_channel(channel: str) -> str:
    value = (channel or "default").strip().lower()
    if value in ("cli", "console", "local", "websocket"):
        return "default"
    if value in ("wechat", "weixin", "wx"):
        return "wechat"
    if value == "feishu":
        return "feishu"
    return "default"


def compute_delivery_backoff_ms(retry_count: int) -> int:
    if retry_count <= 0:
        return 0
    idx = min(retry_count - 1, len(DELIVERY_BACKOFF_MS) - 1)
    base = DELIVERY_BACKOFF_MS[idx]
    jitter = random.randint(-base // 5, base // 5)
    return max(0, base + jitter)


def chunk_message(text: str, channel: str = "default") -> list[str]:
    if not text:
        return []
    ch = normalize_delivery_channel(channel)
    limit = DELIVERY_CHANNEL_LIMITS.get(ch, DELIVERY_CHANNEL_LIMITS["default"])
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    for para in text.split("\n\n"):
        if chunks and len(chunks[-1]) + len(para) + 2 <= limit:
            chunks[-1] += "\n\n" + para
            continue
        while len(para) > limit:
            chunks.append(para[:limit])
            para = para[limit:]
        if para:
            chunks.append(para)
    return chunks or [text[:limit]]


class DeliveryQueue:
    def __init__(self, queue_dir: Path) -> None:
        self.queue_dir = queue_dir
        self.failed_dir = self.queue_dir / "failed"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def enqueue(self, channel: str, to: str, text: str) -> list[str]:
        ids: list[str] = []
        ch = normalize_delivery_channel(channel)
        target = to or "default"
        for chunk in chunk_message(text, ch):
            entry = QueuedDelivery(id=uuid.uuid4().hex[:12], channel=ch, to=target, text=chunk)
            self._write_entry(entry)
            ids.append(entry.id)
        return ids

    def _entry_path(self, delivery_id: str) -> Path:
        return self.queue_dir / f"{delivery_id}.json"

    def _write_entry(self, entry: QueuedDelivery) -> None:
        with self._lock:
            final_path = self._entry_path(entry.id)
            tmp_path = self.queue_dir / f".tmp.{os.getpid()}.{entry.id}.json"
            data = json.dumps(entry.to_dict(), indent=2, ensure_ascii=False)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(str(tmp_path), str(final_path))
            except PermissionError:
                final_path.write_text(data, encoding="utf-8")
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _read_entry(self, delivery_id: str) -> QueuedDelivery | None:
        path = self._entry_path(delivery_id)
        if not path.exists():
            return None
        try:
            return QueuedDelivery.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, OSError, ValueError):
            return None

    def ack(self, delivery_id: str) -> None:
        path = self._entry_path(delivery_id)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {"id": delivery_id}
            data["status"] = "acked"
            data["acked_at"] = time.time()
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def fail(self, delivery_id: str, error: str) -> None:
        entry = self._read_entry(delivery_id)
        if entry is None:
            return
        entry.retry_count += 1
        entry.last_error = error
        if entry.retry_count >= DELIVERY_MAX_RETRIES:
            self.move_to_failed(delivery_id)
            return
        entry.next_retry_at = time.time() + compute_delivery_backoff_ms(entry.retry_count) / 1000.0
        self._write_entry(entry)

    def move_to_failed(self, delivery_id: str) -> None:
        src = self._entry_path(delivery_id)
        dst = self.failed_dir / f"{delivery_id}.json"
        try:
            os.replace(str(src), str(dst))
        except FileNotFoundError:
            pass

    def load_pending(self) -> list[QueuedDelivery]:
        entries: list[QueuedDelivery] = []
        for path in self.queue_dir.glob("*.json"):
            if not path.is_file() or path.name.startswith(".tmp."):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("status") == "acked":
                    continue
                entries.append(QueuedDelivery.from_dict(data))
            except (json.JSONDecodeError, KeyError, OSError, ValueError):
                continue
        entries.sort(key=lambda item: item.enqueued_at)
        return entries

    def load_failed(self) -> list[QueuedDelivery]:
        entries: list[QueuedDelivery] = []
        for path in self.failed_dir.glob("*.json"):
            if not path.is_file() or path.name.startswith(".tmp."):
                continue
            try:
                entries.append(QueuedDelivery.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, KeyError, OSError, ValueError):
                continue
        entries.sort(key=lambda item: item.enqueued_at)
        return entries

    def retry_failed(self) -> int:
        count = 0
        for path in list(self.failed_dir.glob("*.json")):
            try:
                entry = QueuedDelivery.from_dict(json.loads(path.read_text(encoding="utf-8")))
                entry.retry_count = 0
                entry.last_error = ""
                entry.next_retry_at = 0.0
                self._write_entry(entry)
                path.unlink()
                count += 1
            except (json.JSONDecodeError, KeyError, OSError, ValueError):
                continue
        return count


class DeliveryRunner:
    def __init__(
        self,
        delivery_queue: DeliveryQueue,
        channel_mgr: Any,
        command_queue: CommandQueue,
        inbox: Any | None = None,
        notify: Any | None = None,
    ) -> None:
        self.queue = delivery_queue
        self.channel_mgr = channel_mgr
        self.command_queue = command_queue
        self.inbox = inbox
        self.notify = notify
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._inflight_lock = threading.Lock()
        self._inflight: Any | None = None
        self.total_attempted = 0
        self.total_succeeded = 0
        self.total_failed = 0

    def _notify(self, text: str) -> None:
        if self.inbox is not None:
            self.inbox.add("delivery", text)
        elif self.notify is not None:
            self.notify(text)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._notify(f"Recovery: pending={len(self.queue.load_pending())}, failed={len(self.queue.load_failed())}")
        self._thread = threading.Thread(target=self._loop, daemon=True, name="delivery-runner")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

    def request_flush(self, background: bool = False):
        with self._inflight_lock:
            if self._inflight is not None and not self._inflight.done():
                return self._inflight
            self._inflight = self.command_queue.enqueue(
                LANE_DELIVERY,
                lambda: self.process_pending(background=background),
            )
            return self._inflight

    def flush(self, background: bool = False, wait: bool = True, timeout: float | None = None) -> bool:
        future = self.request_flush(background=background)
        if not wait:
            return True
        try:
            future.result(timeout=timeout)
            return True
        except TimeoutError:
            return False

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.request_flush(background=True)
            except Exception as exc:
                self._notify(f"Delivery loop error: {exc}")
            self._stop_event.wait(timeout=1.0)

    def _deliver(self, entry: QueuedDelivery, background: bool = False) -> None:
        if background and entry.channel == "default" and self.inbox is not None:
            self.inbox.add("default", entry.text)
            return
        channel = self.channel_mgr.get(entry.channel)
        if channel is None:
            raise RuntimeError(f"delivery channel '{entry.channel}' is not registered")
        ok = channel.send(entry.to, entry.text)
        if not ok:
            raise RuntimeError(f"delivery channel '{entry.channel}' returned False")

    def process_pending(self, background: bool = False) -> None:
        now = time.time()
        for entry in self.queue.load_pending():
            if self._stop_event.is_set():
                break
            if entry.next_retry_at > now:
                continue
            self.total_attempted += 1
            try:
                self._deliver(entry, background=background)
                self.queue.ack(entry.id)
                self.total_succeeded += 1
            except Exception as exc:
                self.queue.fail(entry.id, str(exc))
                self.total_failed += 1
                retry = entry.retry_count + 1
                if retry >= DELIVERY_MAX_RETRIES:
                    self._notify(f"Delivery {entry.id} moved to failed/: {exc}")
                else:
                    backoff = compute_delivery_backoff_ms(retry)
                    self._notify(
                        f"Delivery {entry.id} failed, retry {retry}/{DELIVERY_MAX_RETRIES} "
                        f"in {backoff / 1000:.0f}s: {exc}"
                    )

    def get_stats(self) -> dict[str, int]:
        return {
            "pending": len(self.queue.load_pending()),
            "failed": len(self.queue.load_failed()),
            "total_attempted": self.total_attempted,
            "total_succeeded": self.total_succeeded,
            "total_failed": self.total_failed,
        }


def enqueue_delivery(delivery_queue: DeliveryQueue | None, channel: str, to: str, text: str) -> bool:
    if delivery_queue is None:
        return False
    return bool(delivery_queue.enqueue(channel, to, text))

