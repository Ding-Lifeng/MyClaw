import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import dashscope

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


# ---------------------------------------------------------------------------
# 核心: Agent 循环
# ---------------------------------------------------------------------------

def agent_loop() -> None:
    messages: list[dict] = []

    print_info("=" * 60)
    print_info("  claw0  |  Section 01: Agent 循环")
    print_info(f"  Model: {MODEL_ID}")
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

        # --- 调用 LLM ---
        try:
            response = dashscope.MultiModalConversation.call(
                api_key = API_KEY,
                model = MODEL_ID,
                messages = messages
            )
        except Exception as exc:
            print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
            messages.pop()
            continue

        # --- 调用终止条件stop_reason ---
        if response.output.choices[0].finish_reason == "stop":
            assistant_text = response.output.choices[0].message.content[0]['text']
            print_assistant(assistant_text)

            messages.append({
                "role" : "assistant",
                "content" : assistant_text,
            })

        elif response.output.choices[0].finish_reason == "tool_calls":
            print_info(f"[finish_reason=tool_calls] 未设置可用工具.")
            print_info("s02继续完善")
            messages.append({
                "role" : "assistant",
                "content" : response.output.choices[0].message.content[0]['text'],
            })

        else:
            print_info(f"[finish_reason]={response.output.choices[0].finish_reason}")
            assistant_text = response.output.choices[0].message.content[0]['text']

            if assistant_text:
                print_assistant(assistant_text)

            messages.append({
                "role" : "assistant",
                "content" : assistant_text,
            })

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