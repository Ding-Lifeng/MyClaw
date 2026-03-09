import os
import sys
from pathlib import Path
import subprocess
from typing import Any
from dotenv import load_dotenv
import dashscope
import json

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
    "Use the tools to help the user with file operations and shell commands.\n"
    "Always read a file before editing it.\n"
    "When using edit_file, the old_string must match EXACTLY (including whitespace)."
)

# 输出字符限制
MAX_TOOL_OUTPUT = 50000

# 工作目录 -- 限制Agent权限
WORKDIR = Path.cwd()

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

# 终端输入提示
def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"

# 输出助手消息
def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")

# 工具调用信息
def print_tool(name: str, detail: str) -> None:
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")

# 输出提示信息
def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")

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

# 截断过长文本
def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} total chars]"

# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------

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
    }
]

# 调度表 - Handler
TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
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
# 核心: Agent 循环
# ---------------------------------------------------------------------------

def agent_loop() -> None:
    messages: list[dict] = []

    print_info("=" * 60)
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Workdir: {WORKDIR}")
    print_info(f"  Tools: {', '.join(TOOL_HANDLERS.keys())}")
    print_info("  输入 'quit' 或 'exit' 退出. Ctrl+C 同样有效.")
    print_info("=" * 60)
    print()

    while True:
        try:
            user_input = input(colored_prompt()).strip()
        except(KeyboardInterrupt, EOFError):
            print(f"\n{DIM}再见.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}再见.{RESET}")
            break

        # --- 添加聊天记录到历史 ---
        messages.append({
            "role" : "user",
            "content" : user_input,
        })

        # --- Agent 工具使用循环 ---
        while True:
            try:
                response = dashscope.MultiModalConversation.call(
                    api_key = API_KEY,
                    model = MODEL_ID,
                    max_tokens = 8096,
                    system = SYSTEM_PROMPT,
                    tools = TOOLS,
                    messages = messages,
                )
            except Exception as exc:
                print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
                while messages and messages[-1]["role"] != "user":
                    messages.pop()
                if messages:
                    messages.pop()
                break

            # print(response) # 响应测试

            if response.status_code != 200:
                print_info(f"\nAPI Error {response.status_code}: {response.message}\n")
                while messages and messages[-1]["role"] != "system":
                    messages.pop()
                break

            choice = response.output.choices[0]
            finish_reason = choice.finish_reason

            # --- 调用终止条件stop_reason ---
            if finish_reason == "stop":
                assistant_text = choice.message.get("content", "")
                if assistant_text:
                    print_assistant(assistant_text)
                messages.append({"role": "assistant", "content": assistant_text})
                break

            elif finish_reason == "tool_calls":
                assistant_text = choice.message
                tool_calls = assistant_text.get("tool_calls", [])

                messages.append(assistant_text) # 添加Assistant的原始消息到历史消息

                for tool_call in tool_calls:
                    print(f"\n{tool_call}\n") # 测试代码
                    function = tool_call["function"]
                    tool_name = function["name"]
                    tool_args = json.loads(function.get("arguments", "{}")) # arguments 是 JSON 格式
                    tool_call_id = tool_call.get("id")

                    result = process_tool_call(tool_name, tool_args) # 工具调用

                    tool_message = {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call_id,
                    }
                    messages.append(tool_message)

                continue

            else:
                print_info(f"[finish_reason]={finish_reason}")
                assistant_text = choice.message.get("content", "")
                if assistant_text:
                    print_assistant(assistant_text)
                    messages.append({"role": "assistant", "content": assistant_text})
                break

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