from __future__ import annotations

from pathlib import Path
from typing import Any

from .intelligence import BootstrapLoader, SkillsManager, build_system_prompt
from .tools import memory_store


def format_memory_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    return "\n".join(
        f"- [{item['path']}] (score: {item['score']}) {item['snippet']}"
        for item in results
    )


class IntelligenceRuntime:
    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.bootstrap_data: dict[str, str] = {}
        self.skills_manager = SkillsManager(workspace_dir)
        self.skills_block = ""

    def refresh(self, mode: str = "full") -> None:
        loader = BootstrapLoader(self.workspace_dir)
        self.bootstrap_data = loader.load_all(mode=mode)
        self.skills_manager.discover()
        self.skills_block = self.skills_manager.format_prompt_block()

    def auto_recall(self, user_message: str, top_k: int = 3) -> str:
        return format_memory_results(memory_store.hybrid_search(user_message, top_k=top_k))

    def compose_system_prompt(
            self,
            agent_id: str,
            channel: str,
            memory_context: str = "",
            mode: str = "full",
    ) -> str:
        return build_system_prompt(
            mode=mode,
            bootstrap=self.bootstrap_data,
            skills_block=self.skills_block,
            memory_context=memory_context,
            agent_id=agent_id,
            channel=channel,
        )
