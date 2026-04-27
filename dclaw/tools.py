from __future__ import annotations

import gzip
import html
import ipaddress
import json
import re
import subprocess
import zlib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from .intelligence import MemoryStore
from .workspace import WorkspacePolicy

# 最大工具输出长度
MAX_TOOL_OUTPUT = 50000
MAX_FETCH_BYTES = 750000
MAX_FETCH_CHARS = 20000
WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36 DClaw/0.1"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
}

# 工作区路径
WORKDIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = WORKDIR / "workspace-main"
WORKSPACE_POLICY = WorkspacePolicy(project_root=WORKDIR, workspace_root=WORKSPACE_DIR)

ShellDecision = Literal["allow", "ask", "deny"]


class ToolMode(Enum):
    SAFE = "safe"
    DEV = "dev"
    TRUSTED = "trusted"

    @staticmethod
    def parse(value: "str | ToolMode") -> "ToolMode":
        if isinstance(value, ToolMode):
            return value
        normalized = (value or "").strip().lower()
        for mode in ToolMode:
            if mode.value == normalized:
                return mode
        raise ValueError(f"Unknown tool mode: {value}")


@dataclass(frozen=True)
class ToolPermissions:
    allow_shell: bool = True
    allow_read: bool = True
    allow_write: bool = True
    allow_edit: bool = True
    allow_memory_write: bool = True
    allow_web_search: bool = True
    allow_web_fetch: bool = True
    allow_fetch_url: bool = True


SAFE_SHELL_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("git", "status"),
    ("git", "diff"),
    ("git", "log"),
    ("python", "--version"),
    ("python", "-V"),
    ("python", "-B", "-m", "unittest"),
    ("python", "-m", "unittest"),
    ("pip", "list"),
    ("pip", "show"),
)

ASK_SHELL_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("git", "commit"),
    ("git", "push"),
    ("pip", "install"),
    ("python",),
    ("pytest",),
    ("npm",),
    ("pnpm",),
    ("uv",),
)


@dataclass
class ToolPolicy:
    mode: ToolMode = ToolMode.DEV
    permissions: ToolPermissions | None = None
    approval_callback: Callable[[str, str], bool] | None = None
    approved_shell_commands: set[str] = field(default_factory=set)
    denied_shell_commands: set[str] = field(default_factory=set)

    def effective_permissions(self) -> ToolPermissions:
        if self.permissions is not None:
            return self.permissions
        if self.mode == ToolMode.SAFE:
            return ToolPermissions(
                allow_shell=False,
                allow_read=True,
                allow_write=False,
                allow_edit=False,
                allow_memory_write=False,
            )
        return ToolPermissions()

    def can_use_tool(self, name: str) -> bool:
        permissions = self.effective_permissions()
        return {
            "bash": permissions.allow_shell,
            "read_file": permissions.allow_read,
            "glob": permissions.allow_read,
            "grep": permissions.allow_read,
            "write_file": permissions.allow_write,
            "edit_file": permissions.allow_edit,
            "list_directory": permissions.allow_read,
            "get_current_time": True,
            "memory_write": permissions.allow_memory_write,
            "memory_search": True,
            "web_search": permissions.allow_web_search,
            "web_fetch": permissions.allow_web_fetch and permissions.allow_fetch_url,
            "fetch_url": permissions.allow_web_fetch and permissions.allow_fetch_url,
        }.get(name, False)

    @staticmethod
    def _matches_prefix(args: list[str], prefix: tuple[str, ...]) -> bool:
        if len(args) < len(prefix):
            return False
        return [part.lower() for part in args[:len(prefix)]] == [part.lower() for part in prefix]

    def classify_shell(self, command: str) -> ShellDecision:
        if not self.can_use_tool("bash"):
            return "deny"
        try:
            args = WORKSPACE_POLICY.parse_shell_command(command)
        except ValueError:
            return "deny"
        if self.mode == ToolMode.SAFE:
            return "deny"
        if self.mode == ToolMode.TRUSTED:
            return "allow"
        if any(self._matches_prefix(args, prefix) for prefix in SAFE_SHELL_PREFIXES):
            return "allow"
        if any(self._matches_prefix(args, prefix) for prefix in ASK_SHELL_PREFIXES):
            return "ask"
        return "ask"

    @staticmethod
    def _approval_key(tool_name: str, detail: str) -> str:
        return f"{tool_name}:{detail.strip()}"

    def approval_status(self, tool_name: str, detail: str) -> ShellDecision | None:
        key = self._approval_key(tool_name, detail)
        if key in self.approved_shell_commands:
            return "allow"
        if key in self.denied_shell_commands:
            return "deny"
        return None

    def approve(self, tool_name: str, detail: str) -> bool:
        key = self._approval_key(tool_name, detail)
        if key in self.approved_shell_commands:
            return True
        if key in self.denied_shell_commands:
            return False
        if self.approval_callback is None:
            self.denied_shell_commands.add(key)
            return False
        approved = bool(self.approval_callback(tool_name, detail))
        if approved:
            self.approved_shell_commands.add(key)
        else:
            self.denied_shell_commands.add(key)
        return approved


TOOL_POLICY = ToolPolicy()


def _default_print_tool(name: str, detail: str) -> None:
    print(f"[tool: {name}] {detail}")

_print_tool = _default_print_tool

def configure_tools(
        print_tool_func=None,
        workspace_policy: WorkspacePolicy | None = None,
        permissions: ToolPermissions | None = None,
        policy: ToolPolicy | None = None,
        mode: ToolMode | str | None = None,
        approval_callback: Callable[[str, str], bool] | None = None,
) -> None:
    global _print_tool, WORKSPACE_POLICY, TOOL_POLICY
    if print_tool_func is not None:
        _print_tool = print_tool_func
    if workspace_policy is not None:
        WORKSPACE_POLICY = workspace_policy
    if policy is not None:
        TOOL_POLICY = policy
        return
    new_mode = TOOL_POLICY.mode if mode is None else ToolMode.parse(mode)
    new_permissions = TOOL_POLICY.permissions
    if permissions is not None:
        new_permissions = permissions
    new_callback = TOOL_POLICY.approval_callback
    if approval_callback is not None:
        new_callback = approval_callback
    TOOL_POLICY = ToolPolicy(
        mode=new_mode,
        permissions=new_permissions,
        approval_callback=new_callback,
        approved_shell_commands=set(TOOL_POLICY.approved_shell_commands),
        denied_shell_commands=set(TOOL_POLICY.denied_shell_commands),
    )

def get_tool_policy() -> ToolPolicy:
    return TOOL_POLICY

def set_tool_mode(mode: ToolMode | str) -> ToolMode:
    global TOOL_POLICY
    parsed = ToolMode.parse(mode)
    TOOL_POLICY = ToolPolicy(
        mode=parsed,
        permissions=None,
        approval_callback=TOOL_POLICY.approval_callback,
        approved_shell_commands=set(TOOL_POLICY.approved_shell_commands),
        denied_shell_commands=set(TOOL_POLICY.denied_shell_commands),
    )
    return parsed

def print_tool(name: str, detail: str) -> None:
    _print_tool(name, detail)

def safe_path(raw: str) -> Path:
    return WORKSPACE_POLICY.resolve_workspace_path(raw)

def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} total chars]"

def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", text)
    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", text)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|li|h[1-6]|tr)>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    text = html.unescape(text)
    body = re.sub(r"[ \t]+", " ", text)
    body = re.sub(r"\n\s*\n+", "\n\n", body).strip()
    if title:
        return f"Title: {title}\n\n{body}"
    return body

def _html_to_markdown(text: str) -> str:
    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", text)
    title = _plain_text_from_html_fragment(title_match.group(1)) if title_match else ""
    body = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", text)
    for level in range(1, 7):
        body = re.sub(
            rf"(?is)<h{level}[^>]*>(.*?)</h{level}>",
            lambda match, level=level: "\n\n" + ("#" * level) + " " + _plain_text_from_html_fragment(match.group(1)) + "\n\n",
            body,
        )
    body = re.sub(
        r'(?is)<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        lambda match: f"[{_plain_text_from_html_fragment(match.group(2))}]({html.unescape(match.group(1))})",
        body,
    )
    body = re.sub(r"(?is)<li[^>]*>(.*?)</li>", lambda match: "\n- " + _plain_text_from_html_fragment(match.group(1)), body)
    body = re.sub(r"(?is)<br\s*/?>", "\n", body)
    body = re.sub(r"(?is)</(p|div|section|article|tr)>", "\n\n", body)
    body = re.sub(r"(?is)<[^>]+>", " ", body)
    body = html.unescape(body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n\s*\n+", "\n\n", body).strip()
    if title and not body.startswith("# "):
        return f"# {title}\n\n{body}".strip()
    return body

def _decode_duckduckgo_url(raw_url: str) -> str:
    value = html.unescape(raw_url or "").strip()
    if value.startswith("//"):
        value = "https:" + value
    parsed = urllib.parse.urlparse(value)
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return query["uddg"][0]
    return value

def _plain_text_from_html_fragment(fragment: str) -> str:
    text = re.sub(r"(?is)<[^>]+>", " ", fragment or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()

def _format_web_candidates(
        candidates: list[tuple[str, str, str]],
        domains: set[str],
        limit: int,
) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for title, result_url, snippet in candidates:
        if result_url in seen:
            continue
        seen.add(result_url)
        host = urllib.parse.urlparse(result_url).netloc.lower()
        if domains and not any(host == domain or host.endswith("." + domain) for domain in domains):
            continue
        clean_title = _plain_text_from_html_fragment(title) or result_url
        clean_snippet = _plain_text_from_html_fragment(snippet) or clean_title
        lines.append(f"- {clean_title}\n  URL: {result_url}\n  Summary: {clean_snippet}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)

def _excerpt(text: str, limit: int = 1200) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "..."

def _decode_http_body(raw: bytes, content_encoding: str) -> bytes:
    encoding = (content_encoding or "").lower()
    if "gzip" in encoding:
        return gzip.decompress(raw)
    if "deflate" in encoding:
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return zlib.decompress(raw, -zlib.MAX_WBITS)
    return raw

def _is_blocked_web_host(hostname: str) -> bool:
    host = (hostname or "").strip().strip("[]").lower().rstrip(".")
    if not host:
        return True
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )

def _validate_web_url(url: str, tool_name: str = "web_fetch") -> str:
    clean_url = (url or "").strip()
    parsed = urllib.parse.urlparse(clean_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{tool_name} only supports absolute http/https URLs.")
    if parsed.username or parsed.password:
        raise ValueError(f"{tool_name} does not support URLs with embedded credentials.")
    if _is_blocked_web_host(parsed.hostname or ""):
        raise ValueError(f"{tool_name} blocked private or internal host: {parsed.hostname or parsed.netloc}")
    return clean_url

def _read_url(url: str, timeout: int, max_bytes: int) -> tuple[str, str, str, bytes]:
    clean_url = _validate_web_url(url)
    request = urllib.request.Request(clean_url, headers=WEB_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            final_url = response.geturl() if hasattr(response, "geturl") else clean_url
            _validate_web_url(final_url)
            content_type = response.headers.get("Content-Type", "")
            content_encoding = response.headers.get("Content-Encoding", "")
            status = str(getattr(response, "status", "") or "")
            raw = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        final_url = exc.geturl() if hasattr(exc, "geturl") else clean_url
        _validate_web_url(final_url)
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        content_encoding = exc.headers.get("Content-Encoding", "") if exc.headers else ""
        status = f"HTTP {exc.code}"
        raw = exc.read(max_bytes + 1)
        if not raw:
            raise
    return final_url, content_type, status, _decode_http_body(raw, content_encoding)

def _parse_duckduckgo_html(html_text: str) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str]] = []
    result_pattern = re.compile(
        r'(?is)<a[^>]+class="[^"]*(?:result__a|result-link)[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
    )
    matches = list(result_pattern.finditer(html_text or ""))
    for index, match in enumerate(matches):
        raw_url = match.group(1)
        title = match.group(2)
        result_url = _decode_duckduckgo_url(raw_url)
        if not result_url.startswith(("http://", "https://")):
            continue
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(html_text)
        block = html_text[match.end():next_start]
        snippet = ""
        snippet_match = re.search(
            r'(?is)<(?:a|div)[^>]+class="[^"]*(?:result__snippet|result-snippet)[^"]*"[^>]*>(.*?)</(?:a|div)>',
            block,
        )
        if snippet_match:
            snippet = snippet_match.group(1)
        candidates.append((title, result_url, snippet or title))
    return candidates

def _format_directory_listing(target: Path, requested: str) -> str:
    entries = sorted(target.iterdir())
    lines = [f"Directory: {target}", f"Requested: {requested or '.'}"]
    if not entries:
        lines.append("[empty directory]")
        return "\n".join(lines)
    for entry in entries:
        prefix = "[dir]  " if entry.is_dir() else "[file] "
        lines.append(prefix + entry.name)
    return "\n".join(lines)

def _shell_directory_listing(command: str) -> str | None:
    try:
        args = WORKSPACE_POLICY.parse_shell_command(command)
    except ValueError:
        return None
    if not args:
        return None
    executable = Path(args[0]).name.lower()
    if executable.endswith(".exe"):
        executable = executable[:-4]
    if executable not in {"dir", "ls"}:
        return None
    if not TOOL_POLICY.can_use_tool("list_directory"):
        return "Error: Directory listing is disabled by tool permissions."

    directory = "."
    for arg in args[1:]:
        normalized = arg.strip()
        if not normalized:
            continue
        if normalized.startswith("-") or normalized.startswith("/"):
            continue
        directory = normalized
        break
    try:
        target = safe_path(directory)
        if not target.exists():
            return f"Error: Directory not found: {directory}"
        if not target.is_dir():
            return f"Error: Not a directory: {directory}"
        return _format_directory_listing(target, directory)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"

# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------
memory_store = MemoryStore(WORKSPACE_DIR)

# shell 命令工具
def tool_bash(command: str, timeout: int = 30) -> str:
    print_tool("bash", command)
    directory_listing = _shell_directory_listing(command)
    if directory_listing is not None:
        return directory_listing
    if not TOOL_POLICY.can_use_tool("bash") and TOOL_POLICY.permissions is not None:
        return "Error: Shell execution is disabled by tool permissions."
    decision = TOOL_POLICY.classify_shell(command)
    if decision == "deny":
        return f"Error: Shell command denied in {TOOL_POLICY.mode.value} mode."
    if decision == "ask":
        previous = TOOL_POLICY.approval_status("bash", command)
        if previous == "deny":
            return f"Error: Shell command was previously denied in {TOOL_POLICY.mode.value} mode."
        if previous != "allow" and not TOOL_POLICY.approve("bash", command):
            return f"Error: Shell command requires approval in {TOOL_POLICY.mode.value} mode."

    try:
        result = WORKSPACE_POLICY.run_shell(command, timeout=timeout)
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
    if not TOOL_POLICY.can_use_tool("read_file"):
        return "Error: File reading is disabled by tool permissions."
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
    if not TOOL_POLICY.can_use_tool("write_file"):
        return "Error: File writing is disabled by tool permissions."
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
    if not TOOL_POLICY.can_use_tool("edit_file"):
        return "Error: File editing is disabled by tool permissions."
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
    if not TOOL_POLICY.can_use_tool("list_directory"):
        return "Error: Directory listing is disabled by tool permissions."
    try:
        target = safe_path(directory)
        if not target.exists():
            return f"Error: Directory not found: {directory}"
        if not target.is_dir():
            return f"Error: Not a directory: {directory}"
        return _format_directory_listing(target, directory)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"

# 文件名匹配工具
def tool_glob(pattern: str, directory: str = ".", max_results: int = 100) -> str:
    print_tool("glob", f"{directory}/{pattern}")
    if not TOOL_POLICY.can_use_tool("glob"):
        return "Error: Glob search is disabled by tool permissions."
    try:
        root = safe_path(directory)
        if not root.exists():
            return f"Error: Directory not found: {directory}"
        if not root.is_dir():
            return f"Error: Not a directory: {directory}"
        clean_pattern = (pattern or "*").strip() or "*"
        if Path(clean_pattern).is_absolute():
            return "Error: Glob pattern must be relative to the workspace directory."
        limit = max(1, min(int(max_results or 100), 500))
        matches: list[str] = []
        for path in sorted(root.glob(clean_pattern)):
            WORKSPACE_POLICY.assert_inside(path, WORKSPACE_POLICY.workspace_root, "glob result")
            rel = path.resolve().relative_to(WORKSPACE_POLICY.workspace_root)
            suffix = "/" if path.is_dir() else ""
            matches.append(str(rel).replace("\\", "/") + suffix)
            if len(matches) >= limit:
                break
        if not matches:
            return f"No files matched pattern: {clean_pattern}"
        return "\n".join(matches)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"

# 文件内容搜索工具
def tool_grep(
        pattern: str,
        directory: str = ".",
        file_glob: str = "**/*",
        case_sensitive: bool = False,
        use_regex: bool = False,
        max_results: int = 100,
) -> str:
    print_tool("grep", f"{pattern} in {directory}/{file_glob}")
    if not TOOL_POLICY.can_use_tool("grep"):
        return "Error: Grep search is disabled by tool permissions."
    try:
        root = safe_path(directory)
        if not root.exists():
            return f"Error: Directory not found: {directory}"
        if not root.is_dir():
            return f"Error: Not a directory: {directory}"
        query = pattern or ""
        if not query:
            return "Error: grep pattern is required."
        clean_glob = (file_glob or "**/*").strip() or "**/*"
        if Path(clean_glob).is_absolute():
            return "Error: file_glob must be relative to the workspace directory."
        limit = max(1, min(int(max_results or 100), 500))
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(query if use_regex else re.escape(query), flags)
        results: list[str] = []
        for path in sorted(root.glob(clean_glob)):
            if not path.is_file():
                continue
            WORKSPACE_POLICY.assert_inside(path, WORKSPACE_POLICY.workspace_root, "grep result")
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = str(path.resolve().relative_to(WORKSPACE_POLICY.workspace_root)).replace("\\", "/")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    snippet = line.strip()
                    if len(snippet) > 240:
                        snippet = snippet[:237] + "..."
                    results.append(f"{rel}:{line_no}: {snippet}")
                    if len(results) >= limit:
                        return "\n".join(results)
        if not results:
            return f"No matches for pattern: {query}"
        return "\n".join(results)
    except re.error as exc:
        return f"Error: Invalid regex: {exc}"
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
def tool_memory_write(content: str, category: str = "general") -> str:
    print_tool("memory_write", f"[{category}] {content[:60]}...")
    if not TOOL_POLICY.can_use_tool("memory_write"):
        return "Error: Memory writing is disabled by tool permissions."
    return memory_store.write_memory(content, category)

# 记忆搜索工具
def tool_memory_search(query: str, top_k: int = 5) -> str:
    print_tool("memory_search", query)
    results = memory_store.hybrid_search(query, top_k)
    if not results:
        return "No relevant memories found."
    return "\n".join(f"[{r['path']}] (score: {r['score']}) {r['snippet']}" for r in results)

# Web 搜索工具
def tool_web_search(
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        recency_days: int | None = None,
) -> str:
    print_tool("web_search", query)
    if not TOOL_POLICY.can_use_tool("web_search"):
        return "Error: Web search is disabled by tool permissions."
    clean_query = (query or "").strip()
    if not clean_query:
        return "Error: web_search query is required."
    if recency_days:
        clean_query = f"{clean_query} recent {recency_days} days"
    limit = max(1, min(int(max_results or 5), 10))
    domains = {domain.lower().lstrip(".") for domain in (allowed_domains or []) if domain}
    html_params = urllib.parse.urlencode({"q": clean_query})
    html_url = f"https://duckduckgo.com/html/?{html_params}"
    try:
        _, _, _, raw = _read_url(html_url, timeout=12, max_bytes=MAX_FETCH_BYTES)
        html_text = raw.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Error: web_search failed: {exc}"

    candidates = _parse_duckduckgo_html(html_text)
    if not candidates:
        return "No web search results found."

    lines: list[str] = []
    seen: set[str] = set()
    for title, result_url, snippet in candidates:
        if result_url in seen:
            continue
        seen.add(result_url)
        host = urllib.parse.urlparse(result_url).netloc.lower()
        if domains and not any(host == domain or host.endswith("." + domain) for domain in domains):
            continue
        clean_title = _plain_text_from_html_fragment(title) or result_url
        clean_snippet = _plain_text_from_html_fragment(snippet) or clean_title
        fetched = tool_web_fetch(result_url, maxChars=3000)
        if fetched.startswith("Error:"):
            fetched_excerpt = f"Page fetch failed ({fetched}). Use the Summary above as the search result snippet."
        else:
            fetched_excerpt = _excerpt(fetched, limit=1200)
        lines.append(
            f"- {clean_title}\n"
            f"  URL: {result_url}\n"
            f"  Summary: {clean_snippet}\n"
            f"  Page excerpt: {fetched_excerpt}"
        )
        if len(lines) >= limit:
            break
    if not lines:
        return "No web search results found."
    return "\n".join(lines)

# URL 抓取工具
def tool_web_fetch(
        url: str,
        maxChars: int | None = None,
        extractMode: str = "text",
        max_chars: int | None = None,
) -> str:
    print_tool("web_fetch", url)
    if not TOOL_POLICY.can_use_tool("web_fetch"):
        return "Error: Web fetching is disabled by tool permissions."
    try:
        clean_url = _validate_web_url(url, "web_fetch")
    except ValueError as exc:
        return f"Error: {exc}"
    requested_limit = maxChars if maxChars is not None else max_chars
    limit = max(1000, min(int(requested_limit or MAX_FETCH_CHARS), MAX_TOOL_OUTPUT))
    mode = (extractMode or "text").strip().lower()
    if mode not in {"text", "markdown", "raw"}:
        return "Error: web_fetch extractMode must be one of: text, markdown, raw."
    try:
        final_url, content_type, status, raw = _read_url(clean_url, timeout=15, max_bytes=MAX_FETCH_BYTES)
    except Exception as exc:
        return f"Error: web_fetch failed: {exc}"
    if len(raw) > MAX_FETCH_BYTES:
        return f"Error: URL response is too large (>{MAX_FETCH_BYTES} bytes)."
    charset = "utf-8"
    match = re.search(r"charset=([\w.-]+)", content_type, flags=re.IGNORECASE)
    if match:
        charset = match.group(1)
    text = raw.decode(charset, errors="replace")
    content_type_l = content_type.lower()
    stripped = text.strip()
    if mode == "raw":
        text = re.sub(r"\r\n?", "\n", text).strip()
    elif "json" in content_type_l or stripped.startswith(("{", "[")):
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            text = re.sub(r"\r\n?", "\n", text).strip()
    elif "html" in content_type_l or re.search(r"(?is)<html|<body|<title", text):
        text = _html_to_markdown(text) if mode == "markdown" else _strip_html(text)
    else:
        text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return f"URL: {final_url}\nContent-Type: {content_type or 'unknown'}\n[empty response]"
    status_line = f"Status: {status}\n" if status else ""
    redirect_line = f"Requested-URL: {clean_url}\n" if final_url != clean_url else ""
    return truncate(
        f"URL: {final_url}\n{redirect_line}{status_line}Content-Type: {content_type or 'unknown'}\n\n{text}",
        limit,
    )

def tool_fetch_url(url: str, max_chars: int = MAX_FETCH_CHARS) -> str:
    return tool_web_fetch(url=url, max_chars=max_chars)

# ---------------------------------------------------------------------------
# 工具定义 - Schema + Handler
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a necessary workspace shell command and return stdout/stderr. "
                "Use this only for tests, project scripts, git status/diff/log, package managers, "
                "or commands that must actually execute. Do not use bash to list directories, "
                "find files, read files, or search text; use list_directory, glob, read_file, or grep instead. "
                "Commands are parsed without shell metacharacters and run inside the tool workspace."
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
            "description": (
                "Read the exact contents of a known file under the workspace. "
                "Use after glob/grep identifies a path, or when the user names a specific file. "
                "Do not use this to check whether many files exist; use glob or list_directory."
            ),
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
                "Write complete content to a workspace file. Creates parent directories if needed "
                "and overwrites existing content. Use for creating new files or replacing an entire file. "
                "For small targeted changes to an existing file, prefer edit_file after read_file."
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
                "Replace one exact string in an existing workspace file with a new string. "
                "The old_string must appear exactly once in the file. "
                "Always read the file first to get exact surrounding text. "
                "Use this for small edits; use write_file only when replacing the whole file."
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
            "description": (
                "List immediate files and subdirectories in a workspace directory. "
                "Use this when the user asks to show the working directory, list files, or inspect a folder. "
                "Returns the resolved workspace path; report only the returned entries. "
                "Do not use bash dir/ls for ordinary directory listing."
            ),
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
            "name": "glob",
            "description": (
                "Find workspace files or directories by path pattern. "
                "Use this when the user asks whether files exist, asks to find files by extension/name, "
                "or before read_file when the exact path is unknown. "
                "Patterns are relative to the workspace, for example '**/*.py' or 'dclaw/**/*.py'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern relative to the directory, such as '**/*.py'.",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Workspace-relative directory to search from. Default '.'.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of paths to return. Default 100, maximum 500.",
                    },
                },
                "required": ["pattern"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search text inside workspace files and return path:line:snippet matches. "
                "Use this for finding code symbols, error messages, TODOs, config keys, or text content. "
                "Do not use bash grep/findstr for normal text search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or regex pattern to search for.",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Workspace-relative directory to search from. Default '.'.",
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Glob limiting files searched, such as '**/*.py'. Default '**/*'.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether matching is case-sensitive. Default false.",
                    },
                    "use_regex": {
                        "type": "boolean",
                        "description": "Treat pattern as a Python regular expression. Default false.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches to return. Default 100, maximum 500.",
                    },
                },
                "required": ["pattern"],
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
            "description": (
                "Save a durable user preference, project fact, or long-term instruction. "
                "Do not use memory to store transient tool results or current file listings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text to remember.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category such as preference, fact, project, context.",
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
            "description": (
                "Search saved long-term memory notes for user preferences or durable facts. "
                "Do not use memory_search to verify current files, directories, or live project state; "
                "use list_directory, glob, grep, or read_file for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of memories to return. Default is 5.",
                    },
                },
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current or external information and return URLs with summaries. "
                "Use for recent facts, third-party documentation, product/news/API changes, or topics not present "
                "in the local workspace. Do not use web_search for local project files; use glob/grep/read_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results. Default 5, maximum 10.",
                    },
                    "allowed_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional domain filter, such as ['docs.python.org'].",
                    },
                    "recency_days": {
                        "type": "integer",
                        "description": "Optional freshness hint in days; appended to the query.",
                    },
                },
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch an http/https URL with a lightweight HTTP GET and return readable page text, markdown, raw text, or formatted JSON with the source URL. "
                "Use after web_search identifies a promising result, especially for pricing pages, docs, "
                "release notes, or pages where the search summary is not enough. "
                "Does not execute JavaScript. Blocks localhost/private/internal hosts. "
                "Do not use web_fetch for local files; use read_file for workspace files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute http or https URL to fetch.",
                    },
                    "maxChars": {
                        "type": "integer",
                        "description": "Maximum characters to return. Default 20000, maximum 50000.",
                    },
                    "extractMode": {
                        "type": "string",
                        "enum": ["text", "markdown", "raw"],
                        "description": "Extraction mode. Use text by default, markdown to preserve headings/links, or raw for non-HTML text.",
                    },
                },
                "required": ["url"],
            },
        }
    }
]

# 调度表 - Handler
TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "glob": tool_glob,
    "grep": tool_grep,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "list_directory": tool_list_directory,
    "get_current_time": tool_get_current_time,
    "memory_write": tool_memory_write,
    "memory_search": tool_memory_search,
    "web_search": tool_web_search,
    "web_fetch": tool_web_fetch,
    "fetch_url": tool_fetch_url,
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
