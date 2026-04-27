from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


SHELL_METACHARS = ("|", "&", ";", ">", "<", "`", "$(", "\n", "\r")
BLOCKED_COMMANDS = {
    "del",
    "erase",
    "format",
    "mkfs",
    "mount",
    "move",
    "rd",
    "reg",
    "ren",
    "rename",
    "rmdir",
    "rm",
    "robocopy",
    "shutdown",
}


@dataclass(frozen=True)
class WorkspacePolicy:
    """Single source of truth for user-visible file and shell boundaries."""

    project_root: Path
    workspace_root: Path
    allow_shell: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", self.project_root.resolve())
        object.__setattr__(self, "workspace_root", self.workspace_root.resolve())
        self.assert_inside(self.workspace_root, self.project_root, "workspace_root")

    @staticmethod
    def assert_inside(path: Path, root: Path, label: str = "path") -> None:
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"{label} resolves outside allowed root: {path}") from exc

    def resolve_workspace_path(self, raw: str) -> Path:
        value = (raw or ".").strip()
        if not value:
            value = "."
        candidate = Path(value)
        if candidate.is_absolute():
            target = candidate.resolve()
        else:
            target = (self.workspace_root / candidate).resolve()
        self.assert_inside(target, self.workspace_root, "workspace path")
        return target

    def parse_shell_command(self, command: str) -> list[str]:
        value = (command or "").strip()
        if not value:
            raise ValueError("Empty shell command.")
        for token in SHELL_METACHARS:
            if token in value:
                raise ValueError(f"Shell metacharacter is not allowed: {token!r}")
        args = shlex.split(value, posix=False)
        if not args:
            raise ValueError("Empty shell command.")
        executable = Path(args[0]).name.lower()
        if executable.endswith(".exe"):
            executable = executable[:-4]
        if executable in BLOCKED_COMMANDS:
            raise ValueError(f"Command is blocked by workspace policy: {args[0]}")
        return args

    def run_shell(self, command: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        if not self.allow_shell:
            raise PermissionError("Shell execution is disabled by workspace policy.")
        args = self.parse_shell_command(command)
        return subprocess.run(
            args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.workspace_root),
        )
