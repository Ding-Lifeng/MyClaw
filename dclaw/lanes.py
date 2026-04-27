from __future__ import annotations

import concurrent.futures
import threading
import time
from collections import deque
from typing import Any, Callable


LANE_MAIN = "main"
LANE_HEARTBEAT = "heartbeat"
LANE_CRON = "cron"
LANE_DELIVERY = "delivery"
DEFAULT_LANE_MAX_CONCURRENCY = 1


class LaneQueue:
    """Named FIFO lane with bounded in-lane concurrency."""

    def __init__(self, name: str, max_concurrency: int = DEFAULT_LANE_MAX_CONCURRENCY) -> None:
        self.name = name
        self.max_concurrency = max(1, max_concurrency)
        self._deque: deque[tuple[Callable[[], Any], concurrent.futures.Future, int]] = deque()
        self._condition = threading.Condition()
        self._active_count = 0
        self._generation = 0

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    def enqueue(self, fn: Callable[[], Any], generation: int | None = None) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._condition:
            gen = self._generation if generation is None else generation
            self._deque.append((fn, future, gen))
            self._pump()
        return future

    def set_max_concurrency(self, value: int) -> None:
        with self._condition:
            self.max_concurrency = max(1, int(value))
            self._pump()
            self._condition.notify_all()

    def reset_generation(self) -> int:
        with self._condition:
            self._generation += 1
            self._condition.notify_all()
            return self._generation

    def _pump(self) -> None:
        while self._active_count < self.max_concurrency and self._deque:
            fn, future, gen = self._deque.popleft()
            if future.set_running_or_notify_cancel():
                self._active_count += 1
                thread = threading.Thread(
                    target=self._run_task,
                    args=(fn, future, gen),
                    daemon=True,
                    name=f"lane-{self.name}",
                )
                thread.start()

    def _run_task(self, fn: Callable[[], Any], future: concurrent.futures.Future, generation: int) -> None:
        try:
            result = fn()
        except Exception as exc:
            self._task_done(generation)
            future.set_exception(exc)
            return
        else:
            self._task_done(generation)
            future.set_result(result)

    def _task_done(self, generation: int) -> None:
        with self._condition:
            self._active_count = max(0, self._active_count - 1)
            if generation == self._generation:
                self._pump()
            self._condition.notify_all()

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        deadline = time.monotonic() + timeout if timeout is not None else None
        with self._condition:
            while self._active_count > 0 or self._deque:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def stats(self) -> dict[str, Any]:
        with self._condition:
            return {
                "name": self.name,
                "queue_depth": len(self._deque),
                "active": self._active_count,
                "max_concurrency": self.max_concurrency,
                "generation": self._generation,
            }


class CommandQueue:
    """Routes callable work into named lanes."""

    def __init__(self) -> None:
        self._lanes: dict[str, LaneQueue] = {}
        self._lock = threading.Lock()

    def get_or_create_lane(self, name: str, max_concurrency: int = DEFAULT_LANE_MAX_CONCURRENCY) -> LaneQueue:
        lane_name = (name or "default").strip().lower()
        with self._lock:
            lane = self._lanes.get(lane_name)
            if lane is None:
                lane = LaneQueue(lane_name, max_concurrency=max_concurrency)
                self._lanes[lane_name] = lane
            return lane

    def enqueue(self, lane_name: str, fn: Callable[[], Any]) -> concurrent.futures.Future:
        return self.get_or_create_lane(lane_name).enqueue(fn)

    def set_max_concurrency(self, lane_name: str, value: int) -> None:
        self.get_or_create_lane(lane_name).set_max_concurrency(value)

    def reset_all(self) -> dict[str, int]:
        with self._lock:
            lanes = list(self._lanes.items())
        return {name: lane.reset_generation() for name, lane in lanes}

    def wait_for_all(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._lock:
            lanes = list(self._lanes.values())
        for lane in lanes:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not lane.wait_for_idle(timeout=remaining):
                return False
        return True

    def stats(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: lane.stats() for name, lane in self._lanes.items()}

    def lane_names(self) -> list[str]:
        with self._lock:
            return sorted(self._lanes)
