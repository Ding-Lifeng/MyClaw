import os, sys, subprocess, json, uuid, time, queue, threading
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import dashscope
from datetime import datetime,timezone
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# ---------------------------------------------------------------------------
# API配置
# ---------------------------------------------------------------------------

# 加载环境变量
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
MODEL_ID = os.getenv("MODEL_ID", "qwen3.5-plus")
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

dashscope.base_http_api_url = BASE_URL

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools.\n"
    "You can also connect to multiple messaging channels.\n"
    "Use tools to help the user with file and time queries.\n"
    "Be concise. If a session has prior context, use it."
)

# 输出字符限制
MAX_TOOL_OUTPUT = 50000

# 上下文限制
CONTEXT_SAFE_LIMIT = 180000

# 工作目录 -- 限制Agent权限
WORKDIR = Path.cwd()

# 状态目录 -- 存储Agent运行过程文件
STATE_DIR = WORKDIR / ".state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ANSI 颜色配置-丰富终端显示效果
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

# 统一输入数据结构
@dataclass
class InboundMessage:
    text: str
    sender_id: str
    channel: str = ""
    account_id: str = ""
    peer_id: str = ""
    is_group: bool = False
    media: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

# 通道帐号设置
@dataclass
class ChannelAccount:
    channel: str
    account_id: str
    token: str = ""
    config: dict = field(default_factory=dict)

# ---------------------------------------------------------------------------
# 实时会话存储
# ---------------------------------------------------------------------------

def build_session_key(channel: str, account_id: str, peer_id: str) -> str:
    return f"agent:main:direct:{channel}:{peer_id}"

# ---------------------------------------------------------------------------
# Channel 类
# ---------------------------------------------------------------------------

# 抽象基类
class Channel(ABC):
    name: str = "unknown"

    @abstractmethod
    def receive(self) -> InboundMessage | None: ...

    @abstractmethod
    def send(self, to: str, text: str, **kwargs: Any) -> bool: ...

    def close(self) -> None:
        pass

# CLI Channel
class CLIChannel(Channel):
    name = "cli"

    def __init__(self) -> None:
        self.account_id = "cli-local"
        self._input_allowed = threading.Event()
        self._input_allowed.set()

    def allow_input(self) -> None:
        self._input_allowed.set()

    def receive(self) -> InboundMessage | None:
        self._input_allowed.wait()
        self._input_allowed.clear()
        try:
            text = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if not text:
            self._input_allowed.set()
            return None
        return InboundMessage(
            text=text, sender_id="cli-user", channel="cli",
            account_id=self.account_id, peer_id="cli-user",
        )

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        print_assistant(text)
        return True

# FeishuChannel - 基于 webhook
class FeishuChannel(Channel):
    name = "feishu"

    def __init__(self, account: ChannelAccount) -> None:
        if not HAS_HTTPX:
            raise RuntimeError("FeishuChannel requires httpx: pip install httpx")
        self.account_id = account.account_id
        self.app_id = account.config.get("app_id", "")
        self.app_secret = account.config.get("app_secret", "")
        self._encrypt_key = account.config.get("encrypt_key", "")
        self._bot_open_id = account.config.get("bot_open_id", "")
        is_lark = account.config.get("is_lark", False)
        self.api_base = ("https://open.larksuite.com/open-apis" if is_lark
                         else "https://open.feishu.cn/open-apis")
        self._tenant_token: str = ""
        self._token_expires_at: float = 0.0
        self._http = httpx.Client(timeout=15.0)

    def _refresh_token(self) -> str:
        if self._tenant_token and time.time() < self._token_expires_at:
            return self._tenant_token
        try:
            resp = self._http.post(
                f"{self.api_base}/auth/v3/tenant_access_token/internal",
                json={"app_id": self.app_id, "app_secret": self.app_secret},
            )
            data = resp.json()
            if data.get("code") != 0:
                print(f"{RED}[feishu] Token error: {data.get('msg', '?')}{RESET}")
                return ""
            self._tenant_token = data.get("tenant_access_token", "")
            self._token_expires_at = time.time() + data.get("expire", 7200) - 300
            return self._tenant_token
        except Exception as exc:
            print(f"{RED}[feishu] Token error: {exc}{RESET}")
            return ""

    def _bot_mentioned(self, event: dict) -> bool:
        for m in event.get("message", {}).get("mentions", []):
            mid = m.get("id", {})
            if isinstance(mid, dict) and mid.get("open_id") == self._bot_open_id:
                return True
            if isinstance(mid, str) and mid == self._bot_open_id:
                return True
            if m.get("key") == self._bot_open_id:
                return True
        return False

    def _parse_content(self, message: dict) -> tuple[str, list]:
        msg_type = message.get("msg_type", "text")
        raw = message.get("content", "{}")
        try:
            content = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            return "", []

        media: list[dict] = []
        if msg_type == "text":
            return content.get("text", ""), media
        if msg_type == "post":
            texts: list[str] = []
            for lc in content.values():
                if not isinstance(lc, dict):
                    continue
                title = lc.get("title", "")
                if title:
                    texts.append(title)
                for para in lc.get("content", []):
                    for node in para:
                        tag = node.get("tag")
                        if tag == "text":
                            texts.append(node.get("text", ""))
                        elif tag == "a":
                            texts.append(node.get("text", "") + " " + node.get("href", ""))
            return "\n".join(texts), media
        if msg_type == "image":
            key = content.get("image_key", "")
            if key:
                media.append({"type": "image", "key": key})
            return "[image]", media
        return "", media

    def parse_event(self, payload: dict, token: str = "") -> InboundMessage | None:
        """解析飞书事件回调。使用简单的 token 校验进行验证。"""
        if self._encrypt_key and token and token != self._encrypt_key:
            print(f"{RED}[feishu] Token verification failed{RESET}")
            return None
        if "challenge" in payload:
            print_info(f"[feishu] Challenge: {payload['challenge']}")
            return None

        event = payload.get("event", {})
        message = event.get("message", {})
        sender = event.get("sender", {}).get("sender_id", {})
        user_id = sender.get("open_id", sender.get("user_id", ""))
        chat_id = message.get("chat_id", "")
        chat_type = message.get("chat_type", "")
        is_group = chat_type == "group"

        if is_group and self._bot_open_id and not self._bot_mentioned(event):
            return None

        text, media = self._parse_content(message)
        if not text:
            return None

        return InboundMessage(
            text=text, sender_id=user_id, channel="feishu",
            account_id=self.account_id,
            peer_id=user_id if chat_type == "p2p" else chat_id,
            media=media, is_group=is_group, raw=payload,
        )

    def receive(self) -> InboundMessage | None:
        return None

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        token = self._refresh_token()
        if not token:
            return False
        try:
            resp = self._http.post(
                f"{self.api_base}/im/v1/messages",
                params={"receive_id_type": "chat_id"},
                headers={"Authorization": f"Bearer {token}"},
                json={"receive_id": to, "msg_type": "text",
                      "content": json.dumps({"text": text})},
            )
            data = resp.json()
            if data.get("code") != 0:
                print(f"{RED}[feishu] Send: {data.get('msg', '?')}{RESET}")
                return False
            return True
        except Exception as exc:
            print(f"{RED}[feishu] Send: {exc}{RESET}")
            return False

    def close(self) -> None:
        self._http.close()

# ---------------------------------------------------------------------------
# Channel 管理
# ---------------------------------------------------------------------------

class ChannelManager:
    def __init__(self) -> None:
        self.channels: dict[str, Channel] = {}
        self.accounts: list[ChannelAccount] = []

    def register(self, channel: Channel) -> None:
        self.channels[channel.name] = channel
        print_channel(f"[+] Channel registered: {channel.name}")

    def list_channels(self) -> list[str]:
        return list(self.channels.keys())

    def get(self, name: str) -> Channel | None:
        return self.channels.get(name)

    def close_all(self) -> None:
        for ch in self.channels.values():
            ch.close()

# ---------------------------------------------------------------------------
# 安全辅助函数
# ---------------------------------------------------------------------------

# Agent的工作路径限制在 WORKDIR 下
def safe_path(raw: str) -> Path:
    target = (WORKDIR / raw).resolve()
    try:
        target.relative_to(WORKDIR.resolve())
    except ValueError:
        raise ValueError(f"Path traversal blocker: {raw} resolves outside WORKDIR")
    return target

# ---------------------------------------------------------------------------
# 会话存储 -- 基于 JSONL
# ---------------------------------------------------------------------------

class SessionStore:
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.base_dir = WORKDIR / ".sessions" / "agents" / agent_id / "sessions"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "session.json"
        self._index: dict[str, dict] = self._load_index()
        self.current_session_id: str | None = None

    def _load_index(self) -> dict[str, dict]:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _session_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.jsonl"

    def create_session(self, label: str = "") -> str:
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        self._index[session_id] = {
            "label": label,
            "created_at": now,
            "last_active": now,
            "message_count": 0,
        }
        self._save_index()
        self._session_path(session_id).touch()
        self.current_session_id = session_id
        return session_id

    # 加载会话记录
    def load_session(self, session_id: str) -> list[dict]:
        path = self._session_path(session_id)
        if not path.exists():
            return []
        self.current_session_id = session_id
        return self._rebuild_history(path)

    def append_transcript(self, session_id: str, record: dict) -> None:
        path = self._session_path(session_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if session_id in self._index:
            self._index[session_id]["last_active"] = (
                datetime.now(timezone.utc).isoformat()
            )
            self._index[session_id]["message_count"] += 1
            self._save_index()

    # 根据 JSONL 重建 messages
    @staticmethod
    def _rebuild_history(path: Path) -> list[dict]:
        messages: list[dict] = []

        if not path.exists():
            return messages

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "role" in msg:
                messages.append(msg)
        return messages

    def list_sessions(self) -> list[tuple[str, dict]]:
        items = list(self._index.items())
        items.sort(key=lambda x: x[1].get("last_active", ""), reverse=True)
        return items

# 精简消息格式 - LLM 生成摘要
def _serialize_messages_for_summary(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(f"[{role}]: {block.get('text', '')}")
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# 处理会话消息-防止上下文溢出
# ---------------------------------------------------------------------------

class ContextGuard:
    def __init__(self, max_tokens: int = CONTEXT_SAFE_LIMIT):
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    # 估算消息的Token消耗
    def estimate_messages_tokens(self, messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.estimate_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if "text" in block:
                            total += self.estimate_tokens(block["text"])
                        elif block.get("type") == "tool_result":
                            rc = block.get("content", "")
                            if isinstance(rc, str):
                                total += self.estimate_tokens(rc)
                        elif block.get("type") == "tool_use":
                            total += self.estimate_tokens(
                                json.dumps(block.get("input", {}))
                            )
                    else:
                        if hasattr(block, "text"):
                            total += self.estimate_tokens(block.text)
                        elif hasattr(block, "input"):
                            total += self.estimate_tokens(
                                json.dumps(block.input)
                            )
        return total

    # 截断文本 - 并尽量保持可读性
    def truncate_tool_result(self, result: str, max_fraction: float = 0.3) -> str:
        max_chars = int(self.max_tokens * 4 * max_fraction)
        if len(result) <= max_chars:
            return result
        cut = result.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        head = result[:cut]
        return head + f"\n\n[... truncated ({len(result)} chars total, showing first {len(head)}) ...]"

    # 总结历史消息 - 形成摘要
    @staticmethod
    def compact_history(messages: list[dict]) -> list[dict]:
        total = len(messages)
        if total <= 4:
            return messages

        keep_count = max(4, int(total * 0.2))
        compress_count = max(2, int(total * 0.5))
        compress_count = min(compress_count, total - keep_count)

        if compress_count < 2:
            return messages

        old_messages = messages[:compress_count]
        recent_messages = messages[compress_count:]

        old_text = _serialize_messages_for_summary(old_messages)

        summary_prompt = (
            "Summarize the following conversation concisely, "
            "preserving key facts and decisions. "
            "Output only the summary, no preamble.\n\n"
            f"{old_text}"
        )

        try:
            response = dashscope.MultiModalConversation.call(
                api_key=API_KEY,
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "You are a conversation summarizer. Be concise and factual."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=2048,
                result_format='message'
            )
            if response.status_code != 200:
                raise Exception(f"API Error {response.status_code}: {response.message}")

            content = response.output.content[0].content
            if isinstance(content, list):
                summary_text = "\n".join(item["text"] for item in content if isinstance(item, dict) and "text" in item)
            else:
                summary_text = str(content)

            print_session(
                f" [compact] {len(old_messages)} messages -> summary "
                f"({len(summary_text)} chars)"
            )
        except Exception as exc:
            print_warn(f" [compact] Summary failed ({exc}), dropping old messages")
            return recent_messages

        compacted = [
            {
                "role": "user",
                "content": "[Previous conversation summary]\n" + summary_text,
            },
            {
                "role": "assistant",
                "content": "Understood, I have the context from our previous conversation."  # 直接使用字符串
            },
        ]

        compacted.extend(recent_messages)
        return compacted

    # 遍历消息列表 - 截断过长的工具调用结果
    def _truncate_large_tool_results(self, messages: list[dict]) -> list[dict]:
        result = []
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content")
                if isinstance(content, str):
                    new_msg = msg.copy()
                    new_msg["content"] = self.truncate_tool_result(content)
                    result.append(new_msg)
                else:
                    result.append(msg)
            else:
                result.append(msg)
        return result

    def guard_api_call(
            self,
            api_key: str,
            model: str,
            system: str,
            messages: list[dict],
            max_tokens: int = 8096,
            tools: list[dict] | None = None,
            max_retries: int = 2,
    ) -> Any:
        current_messages = messages

        for attempt in range(max_retries + 1):
            try:
                full_messages = [{"role": "system", "content": system}] + current_messages

                kwargs = {
                    "api_key": api_key,
                    "model": model,
                    "max_tokens": 8096,
                    "messages": full_messages,
                    "result_format": "message",
                }
                if tools:
                    kwargs["tools"] = tools

                response = dashscope.MultiModalConversation.call(**kwargs)

                if response.status_code != 200:
                    raise Exception(f"API Error {response.status_code}: {response.message}")

                if current_messages is not messages:
                    messages.clear()
                    messages.extend(current_messages)
                return response

            except Exception as exc:
                error_str = str(exc).lower()
                is_overflow = "context" in error_str or "token" in error_str

                if not is_overflow or attempt >= max_retries:
                    raise

                if attempt == 0:
                    print_warn(
                        "  [guard] Context overflow detected, "
                        "truncating large tool results..."
                    )
                    current_messages = self._truncate_large_tool_results(current_messages)
                elif attempt == 1:
                    print_warn(
                        "  [guard] Still overflowing, "
                        "compacting conversation history..."
                    )
                    current_messages = self.compact_history(current_messages)

        raise RuntimeError("guard_api_call: exhausted retries")

# 截断过长文本 TODO:统一truncate工具
def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} total chars]"

# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------
MEMORY_FILE = WORKDIR / "MEMORY.md"

# shell 命令工具
def tool_bash(command: str, timeout: int = 30) -> str:
    dangerous = ["rm -rf /", "mkfs", "> /dev/sd", "dd if="] # 拒绝危险命令
    for pattern in dangerous:
        if pattern in command:
            return f"Error: Refused to run dangerous command containing '{pattern}'"

    print_tool("bash", command)

    try:
        result = subprocess.run(
            command,
            shell = True,
            capture_output = True,
            text = True,
            timeout = timeout,
            cwd = str(WORKDIR),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr)
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return truncate(output) if output else "[no output]"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as exc:
        return f"Error: {exc}"

# 读文件工具
def tool_read_file(file_path: str) -> str:
    print_tool("read_file", file_path)
    try:
        target = safe_path(file_path) # 检查工作路径
        if not target.exists():
            return f"Error: File not found: {file_path}"
        if not target.is_file():
            return f"Error:Not a file: {file_path}"
        content = target.read_text(encoding="utf-8")
        return truncate(content)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"

# 写文件工具
def tool_write_file(file_path: str, content: str) -> str:
    print_tool("write_file", file_path)
    try:
        target = safe_path(file_path) # 检查工作路径
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} chars to {file_path}"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"

# 编辑文件工具
def tool_edit_file(file_path: str, old_string: str, new_string: str) -> str:
    print_tool("edit_file", f"{file_path} (replace {len(old_string)} chars)")
    try:
        target = safe_path(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"

        content = target.read_text(encoding="utf-8")
        count = content.count(old_string)

        if count == 0:
            return "Error: old_string not found in file. Make sure it matches exactly."
        if count > 1:
            return (
                f"Error: old_string found {count} times. "
                "It must be unique. Provide more surrounding context."
            )

        new_content = content.replace(old_string, new_string, 1)
        target.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {file_path}"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"

# 列出目录内容工具
def tool_list_directory(directory: str = ".") -> str:
    print_tool("list_directory", directory)
    try:
        target = safe_path(directory)
        if not target.exists():
            return f"Error: Directory not found: {directory}"
        if not target.is_dir():
            return f"Error: Not a directory: {directory}"
        entries = sorted(target.iterdir())
        lines = []
        for entry in entries:
            prefix = "[dir]  " if entry.is_dir() else "[file] "
            lines.append(prefix + entry.name)
        return "\n".join(lines) if lines else "[empty directory]"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"

# 获取时间工具
def tool_get_current_time() -> str:
    print_tool("get_current_time", "")
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")

# 记忆书写工具
def tool_memory_write(content: str) -> str:
    print_tool("memory_write", f"{len(content)} chars")
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n- {content}\n")
        return f"Written to memory: {content[:80]}..."
    except Exception as exc:
        return f"Error: {exc}"

# 记忆搜索工具
def tool_memory_search(query: str) -> str:
    print_tool("memory_search", query)
    if not MEMORY_FILE.exists():
        return "Memory file is empty."
    try:
        lines = MEMORY_FILE.read_text(encoding="utf-8").splitlines()
        matches = [l for l in lines if query.lower() in l.lower()]
        return "\n".join(matches[:20]) if matches else f"No matches for '{query}'."
    except Exception as exc:
        return f"Error: {exc}"

# ---------------------------------------------------------------------------
# 工具定义 - Schema + Handler
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command and return its output. "
                "Use for system commands, git, package managers, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds. Default 30.",
                    },
                },
                "required": ["command"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file (relative to working directory).",
                    }
                },
                "required": ["file_path"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Wrote content to a file. Creates parent directories if needed. "
                "Overwrites existing content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file (relative to working directory).",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write.",
                    }
                },
                "required": ["file_path", "content"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace an exact string in a file with a new string. "
                "The old_string must appear exactly once in the file. "
                "Always read the file first to get the exact text to replace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file (relative to working directory).",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find and replace. Must be unique.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text.",
                    }
                },
                "required": ["file_path", "old_string", "new_string"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories in a directory under workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Path relative to workspace directory. Default is root.",
                    },
                },
                "required": [],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time in UTC.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Save a note to long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text to remember.",
                    },
                },
                "required": ["content"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search through saved memory notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword.",
                    },
                },
                "required": ["query"],
            },
        }
    }
]

# 调度表 - Handler
TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "list_directory": tool_list_directory,
    "get_current_time": tool_get_current_time,
    "memory_write": tool_memory_write,
    "memory_search": tool_memory_search,
}

# ---------------------------------------------------------------------------
# 工具调用
# ---------------------------------------------------------------------------

def process_tool_call(tool_name: str, tool_input: dict[str, Any]) -> str:
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"Error: Unknown tool: {tool_name}"
    try:
        return handler(**tool_input)
    except TypeError as exc:
        return f"Error: Invalid arguments for {tool_name}: {exc}"
    except Exception as exc:
        return f"Error: {tool_name} failed: {exc}"

# ---------------------------------------------------------------------------
# Read-Eval-Print Loop
# ---------------------------------------------------------------------------

def handle_repl_command(
        command: str,
        store: SessionStore,
        guard: ContextGuard,
        messages: list[dict],
        mgr: ChannelManager
) -> tuple[bool, list[dict]]:
    parts = command.strip().split(maxsplit = 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/new":
        label = arg or ""
        sid = store.create_session(label)
        print_session(f"Created new session: {sid}" + (f" ({label})" if label else ""))
        return True,[]
    elif cmd == "/list":
        sessions = store.list_sessions()
        if not sessions:
            print_info("No sessions found.")
            return True, messages

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
        return True, messages

    elif cmd == "/switch":
        if not arg:
            print_warn("Usage: /switch <session_id>")
            return True,messages
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
            return True, messages
        if len(matched) > 1:
            print_warn(f"Ambiguous prefix, matches: {', '.join(matched)}")
            return True, messages

        new_sid = matched[0]
        new_messages = store.load_session(new_sid)
        print_session(f" Switched to session: {new_sid} ({len(new_messages)} messages)")
        return True, new_messages

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
        return True, messages

    elif cmd == "/compact":
        if len(messages) <= 4:
            print_info("Too few messages to compact (need > 4).")
            return True, messages
        print_session("Compacting history...")
        new_messages = guard.compact_history(messages)
        print_session(f"{len(messages)} -> {len(new_messages)} messages")
        return True, new_messages

    elif cmd == "/channels":
        channels = mgr.list_channels()
        if channels:
            print_channel("Channels:")
            for name in channels:
                print_channel(f"  - {name}")
        else:
            print_info("No channels.")
        return True, messages

    elif cmd == "/accounts":
        accounts = mgr.accounts
        if accounts:
            print_channel("Accounts:")
            for acc in accounts:
                masked = acc.token[:8] + "..." if len(acc.token) > 8 else "(none)"
                print_channel(f"- {acc.channel}/{acc.account_id}  token={masked}")
        else:
            print_info("No accounts.")
        return True, messages

    elif cmd == "/help":
        print_info("  Commands:")
        print_info("    /new [label]       Create a new session")
        print_info("    /list              List all sessions")
        print_info("    /switch <id>       Switch to a session (prefix match)")
        print_info("    /context           Show context token usage")
        print_info("    /compact           Manually compact conversation history")
        print_info("    /channels          List all active channels")
        print_info("    /accounts          List configured bot accounts")
        print_info("    /help              Show this help")
        print_info("    quit / exit        Exit the REPL")
        return True, messages

    return False, messages

# ---------------------------------------------------------------------------
# 辅助函数 - 格式处理
# ---------------------------------------------------------------------------

# 从模型的返回值中提取文本
def extract_assistant_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
        return "\n".join(texts)
    return str(content)

# ---------------------------------------------------------------------------
# Agent 交互回合
# ---------------------------------------------------------------------------

def run_agent_turn(
        inbound: InboundMessage,
        conversations: dict[str, list[dict]],
        mgr: ChannelManager,
        store: SessionStore | None = None,
) -> None:
    sk = build_session_key(inbound.channel, inbound.account_id, inbound.peer_id)
    if sk not in conversations:
        conversations[sk] = []
    messages = conversations[sk]

    should_presist = (store is not None and inbound.channel == "cli")

    # --- 添加聊天记录到历史 ---
    user_message = {
        "role": "user",
        "content": inbound.text,
    }
    messages.append(user_message)
    if should_presist:
        user_record = user_message.copy()
        store.append_transcript(store.current_session_id, user_record)

    guard = ContextGuard()

    while True:
        try:
            response = guard.guard_api_call(
                api_key=API_KEY,
                model=MODEL_ID,
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as exc:
            print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
            while messages and messages[-1]["role"] != "user":
                messages.pop()
            if messages:
                messages.pop()
            break

        if response.status_code != 200:
            print_info(f"\nAPI Error {response.status_code}: {response.message}\n")
            while messages and messages[-1]["role"] != "system":
                messages.pop()
            break

        choice = response.output.choices[0]
        finish_reason = choice.finish_reason
        assistant_message = choice.message

        assistant_dict = {
            "role": "assistant",
            "content": assistant_message.content if hasattr(assistant_message, "content") else None
        }
        tool_calls = assistant_message.get("tool_calls" if isinstance(assistant_message, dict) else None)
        if tool_calls:
            assistant_dict["tool_calls"] = assistant_message.tool_calls
        messages.append(assistant_dict)
        if should_presist:
            assistant_record = assistant_dict.copy()
            store.append_transcript(store.current_session_id, assistant_record)

        # --- 调用终止条件stop_reason ---
        if finish_reason == "stop":
            text = extract_assistant_text(assistant_message)
            if text:
                ch = mgr.get(inbound.channel)
                if ch:
                    ch.send(inbound.peer_id, text)
                else:
                    print_assistant(text)
            break

        elif finish_reason == "tool_calls":
            tool_calls = assistant_message.get("tool_calls", [])

            for tool_call in tool_calls:
                # print(f"\n{tool_call}\n") # 测试代码
                function = tool_call["function"]
                tool_name = function["name"]
                tool_args = json.loads(function.get("arguments", "{}"))  # arguments 是 JSON 格式
                tool_call_id = tool_call.get("id")

                result = process_tool_call(tool_name, tool_args)  # 工具调用

                tool_message = {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call_id,
                }
                messages.append(tool_message)
                if should_presist:
                    tool_record = tool_message.copy()
                    store.append_transcript(store.current_session_id, tool_record)

            continue

        else:
            print_info(f"[finish_reason]={finish_reason}")
            text = extract_assistant_text(assistant_message)
            if text:
                ch = mgr.get(inbound.channel)
                if ch:
                    ch.send(inbound.peer_id, text)
                else:
                    print_assistant(text)
            break

# ---------------------------------------------------------------------------
# 核心: Agent 循环
# ---------------------------------------------------------------------------

def agent_loop() -> None:
    mgr = ChannelManager()  # 初始化通道管理

    # 注册 CLI 通道
    cli = CLIChannel()
    mgr.register(cli)

    # 注册飞书通道
    fs_id = os.getenv("FEISHU_APP_ID", "").strip()
    fs_secret = os.getenv("FEISHU_APP_SECRET", "").strip()
    if fs_id and fs_secret and HAS_HTTPX:
        fs_acc = ChannelAccount(
            channel="feishu",
            account_id="feishu-primary",
            config={
                "app_id": fs_id,
                "app_secret": fs_secret,
                "encrypt_key": os.getenv("FEISHU_ENCRYPT_KEY", ""),
                "bot_open_id": os.getenv("FEISHU_BOT_OPEN_ID", ""),
                "is_lark": os.getenv("FEISHU_IS_LARK", "").lower() in ("1", "true"),
            }
        )
        mgr.accounts.append(fs_acc)
        mgr.register(FeishuChannel(fs_acc))
        print_channel("[+] Feishu channel registered (requires webhook server)")

    # CLI 持久化
    store = SessionStore(agent_id="MyClaw")
    guard = ContextGuard()

    # 内存会话存储 - Channel实时对话
    conversations: dict[str, list[dict]] = {}

    cli_sk = build_session_key("cli", "cli-local", "cli-user")
    sessions = store.list_sessions()
    if sessions:
        sid = sessions[0][0]
        cli_history = store.load_session(sid)
        conversations[cli_sk] = cli_history
        print_session(f"Resumed session: {sid} ({len(cli_history)} messages)")
    else:
        sid = store.create_session("initial")
        conversations[cli_sk] = []
        print_session(f"Created initial session: {sid}")

    msg_queue: queue.Queue[InboundMessage | None] = queue.Queue()
    stop_event = threading.Event()

    def cli_reader():
        while not stop_event.is_set():
            msg = cli.receive()
            if msg is None:
                continue
            msg_queue.put(msg)
        print_info("CLI reader stopped.")

    print_info("=" * 60)
    print_info(f" Model: {MODEL_ID}")
    print_info(f" Session: {store.current_session_id}")
    print_info(f" Channels: {', '.join(mgr.list_channels())}")
    print_info(f" Workdir: {WORKDIR}")
    print_info(f" Tools: {', '.join(TOOL_HANDLERS.keys())}")
    print_info("  输入 /help 获取指令提示, 输入 'quit' 或 'exit' 退出.")
    print_info("=" * 60)
    print()

    cli_thread = threading.Thread(target=cli_reader, daemon=True)
    cli_thread.start()

    try:
        while not stop_event.is_set():
            try:
                msg = msg_queue.get(timeout=0.5)  # 从消息队列中读取不同通道消息
            except queue.Empty:
                continue

            if msg is None:
                break

            if msg.channel == "cli":
                if msg.text.lower() in ("quit", "exit"):
                    stop_event.set()
                    break

                if msg.text.startswith("/"):
                    current_messages = conversations.get(cli_sk, [])
                    handled, new_messages = handle_repl_command(
                        msg.text, store, guard, current_messages, mgr
                    )
                    if handled:
                        conversations[cli_sk] = new_messages
                        cli.allow_input()
                        continue

            if msg.channel == "cli":
                run_agent_turn(msg, conversations, mgr, store=store)
                cli.allow_input()
            else:
                run_agent_turn(msg, conversations, mgr)

    except KeyboardInterrupt:
        print(f"")
    finally:
        stop_event.set()
        cli_thread.join(timeout=2.0)
        mgr.close_all()
        print(f"{DIM}再见.{RESET}")

# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.getenv("DASHSCOPE_API_KEY"):
        print(f"{YELLOW}Error: DASHSCOPE_API_KEY 未设置.{RESET}")
        print(f"{DIM}环境配置未完成!{RESET}")
        sys.exit(1)

    agent_loop()

if __name__ == "__main__":
    main()