from __future__ import annotations

import json

from .llm import LLMClient
from .runtime import _serialize_messages_for_summary

CONTEXT_SAFE_LIMIT = 180000

class ContextGuard:
    def __init__(self, max_tokens: int = CONTEXT_SAFE_LIMIT, print_session_func=None, print_warn_func=None):
        self.max_tokens = max_tokens
        self._print_session = print_session_func or print
        self._print_warn = print_warn_func or print

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
    def compact_history(self, messages: list[dict], client: LLMClient, model: str) -> list[dict]:
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
            response = client.chat(
                model = model,
                system = "You are a conversation summarizer. Be concise and factual.",
                messages = [{"role": "user", "content": summary_prompt}],
                max_tokens = 2048,
                tools = None
            )

            summary_text = response["text"]

            self._print_session(
                f" [compact] {len(old_messages)} messages -> summary "
                f"({len(summary_text)} chars)"
            )
        except Exception as exc:
            self._print_warn(f" [compact] Summary failed ({exc}), dropping old messages")
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
            role = msg.get("role")
            content = msg.get("content")

            if role == "tool" and isinstance(content, str):
                new_msg = msg.copy()
                new_msg["content"] = self.truncate_tool_result(content)
                result.append(new_msg)
                continue

            if role == "user" and isinstance(content, list):
                new_blocks = []
                changed = False
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and isinstance(block.get("content"), str)
                    ):
                        new_block = block.copy()
                        new_block["content"] = self.truncate_tool_result(block["content"])
                        new_blocks.append(new_block)
                        changed = True
                    else:
                        new_blocks.append(block)
                if changed:
                    new_msg = msg.copy()
                    new_msg["content"] = new_blocks
                    result.append(new_msg)
                else:
                    result.append(msg)
                continue

            result.append(msg)
        return result

    def guard_api_call(
            self,
            client: LLMClient,
            model: str,
            system: str,
            messages: list[dict],
            max_tokens: int = 8096,
            tools: list[dict] | None = None,
            max_retries: int = 2,
    ) -> dict:
        current_messages = messages.copy()

        for attempt in range(max_retries + 1):
            try:
                response = client.chat(
                    model = model,
                    system = system,
                    messages = current_messages,
                    max_tokens = max_tokens,
                    tools = tools,
                )

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
                    self._print_warn(
                        "  [guard] Context overflow detected, "
                        "truncating large tool results..."
                    )
                    current_messages = self._truncate_large_tool_results(current_messages)
                elif attempt == 1:
                    self._print_warn(
                        "  [guard] Still overflowing, "
                        "compacting conversation history..."
                    )
                    current_messages = self.compact_history(current_messages, client, model)

        raise RuntimeError("guard_api_call: exhausted retries")
