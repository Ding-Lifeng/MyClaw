from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any


CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"

# 终端输入提示
def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"

# 输出助手消息
def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")

# 工具调用信息
def print_tool(name: str, detail: str) -> None:
    print(f"{DIM}[tool: {name}] {detail}{RESET}")

# 输出提示信息
def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")

# 输出警告信息
def print_warn(text: str) -> None:
    print(f"{YELLOW}{text}{RESET}")

# 输出会话信息
def print_session(text: str) -> None:
    print(f"{MAGENTA}{text}{RESET}")

# 输出通道信息
def print_channel(text: str) -> None:
    print(f"{BLUE}{text}{RESET}")

def print_section(title: str) -> None:
    print(f"\n{MAGENTA}{BOLD}--- {title} ---{RESET}")

def print_heartbeat(text: str) -> None:
    print(f"{BLUE}{BOLD}[heartbeat]{RESET} {text}")

def print_cron(text: str) -> None:
    print(f"{MAGENTA}{BOLD}[cron]{RESET} {text}")

def print_delivery(text: str) -> None:
    print(f"{BLUE}{BOLD}[delivery]{RESET} {text}")

class BackgroundInbox:
    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, source: str, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._items.append({
                "source": source,
                "text": text,
                "ts": datetime.now(timezone.utc).isoformat(),
            })

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def list_items(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._items)

    def clear(self) -> int:
        with self._lock:
            count = len(self._items)
            self._items.clear()
            return count
