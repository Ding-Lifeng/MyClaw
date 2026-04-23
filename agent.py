import os, sys, subprocess, json, uuid, time, asyncio, hashlib
import math, queue, threading, re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional
from dotenv import load_dotenv
import dashscope
from datetime import datetime, timezone
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

MODEL_ID = os.getenv("MODEL_ID", "MiniMax-M2.7")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").lower()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

dashscope.base_http_api_url = DASHSCOPE_BASE_URL

# ---------------------------------------------------------------------------
# 系统配置
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools.\n"
    "You can also connect to multiple messaging channels.\n"
    "Use tools to help the user with file and time queries.\n"
    "Be concise. If a session has prior context, use it."
)

# Agent 启动加载文件
BOOTSTRAP_FILES = [
    "SOUL.md", "IDENTITY.md", "TOOLS.md", "USER.md",
    "HEARTBEAT.md", "BOOTSTRAP.md", "AGENTS.md", "MEMORY.md",
]

MAX_FILE_CHARS = 20000
MAX_TOTAL_CHARS = 150000
MAX_SKILLS = 150
MAX_SKILLS_PROMPT = 30000
LOCAL_EMBED_DIM = 256
LOCAL_EMBED_PROJECTIONS = 6
LOCAL_EMBED_TOKEN_WEIGHT = 1.0
LOCAL_EMBED_BIGRAM_WEIGHT = 0.7
LOCAL_EMBED_TRIGRAM_WEIGHT = 0.9
LOCAL_EMBED_PATH_WEIGHT = 1.2
LOCAL_EMBED_PATH_PART_WEIGHT = 1.0
LOCAL_EMBED_CATEGORY_WEIGHT = 1.2
LOCAL_EMBED_CACHE_MAX = 512

# 输出字符限制
MAX_TOOL_OUTPUT = 50000

# 上下文限制
CONTEXT_SAFE_LIMIT = 180000

# 工作目录 -- 限制Agent权限
WORKDIR = Path.cwd()
WORKSPACE_DIR = WORKDIR / "workspace-main"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Agents 目录 -- 多Agent
AGENTS_DIR = WORKDIR / ".agents"

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

def print_section(title: str) -> None:
    print(f"\n{MAGENTA}{BOLD}--- {title} ---{RESET}")

# ---------------------------------------------------------------------------
# Bootstrap 文件加载
# ---------------------------------------------------------------------------

class BootstrapLoader:

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir

    def load_file(self, name: str) -> str:
        path = self.workspace_dir / name
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    @staticmethod
    def truncate_file(content: str, max_chars: int = MAX_FILE_CHARS) -> str:
        if len(content) <= max_chars:
            return content
        cut = content.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        return content[:cut] + f"\n\n[... truncated ({len(content)} chars total, showing first {cut}) ...]"

    def load_all(self, mode: str = "full") -> dict[str, str]:
        if mode == "none":
            return {}
        names = ["AGENTS.md", "TOOLS.md"] if mode == "minimal" else list(BOOTSTRAP_FILES)
        result: dict[str, str] = {}
        total = 0
        for name in names:
            raw = self.load_file(name)
            if not raw:
                continue
            truncated = self.truncate_file(raw)
            if total + len(truncated) > MAX_TOTAL_CHARS:
                remaining = MAX_TOTAL_CHARS - total
                if remaining <= 0:
                    break
                truncated = self.truncate_file(raw, remaining)
            result[name] = truncated
            total += len(truncated)
        return result

# ---------------------------------------------------------------------------
# SOUL
# ---------------------------------------------------------------------------

def load_soul(workspace_dir: Path) -> str:
    path = workspace_dir / "SOUL.md"
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

class SkillsManager:

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.skills: list[dict[str, str]] = []

    @staticmethod
    def _parse_frontmatter(text: str) -> dict[str, str]:
        meta: dict[str, str] = {}
        if not text.startswith("---"):
            return meta
        parts = text.split("---", 2)
        if len(parts) < 3:
            return meta
        for line in parts[1].strip().splitlines():
            if ":" not in line:
                continue
            key, _, value = line.strip().partition(":")
            meta[key.strip()] = value.strip().strip('"').strip("'")
        return meta

    def _scan_dir(self, base: Path) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        if not base.is_dir():
            return found
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                content = skill_md.read_text(encoding="utf-8")
            except Exception:
                continue
            meta = self._parse_frontmatter(content)
            body = content
            if content.startswith("---"):
                parts = content.split("---", 2)
                body = parts[2].strip() if len(parts) >= 3 else ""
            name = meta.get("name") or child.name
            found.append({
                "name": name,
                "description": meta.get("description", ""),
                "invocation": meta.get("invocation", f"Use when the task matches {name}."),
                "body": body,
                "path": str(skill_md),
            })
        return found

    def discover(self, extra_dirs: list[Path] | None = None) -> None:
        scan_order: list[Path] = []
        if extra_dirs:
            scan_order.extend(extra_dirs)
        scan_order.extend([
            self.workspace_dir / "skills",
            self.workspace_dir / ".skills",
            self.workspace_dir / ".agents" / "skills",
            WORKDIR / ".agents" / "skills",
            WORKDIR / "skills",
        ])
        seen: dict[str, dict[str, str]] = {}
        for directory in scan_order:
            for skill in self._scan_dir(directory):
                seen[skill["name"]] = skill
        self.skills = list(seen.values())[:MAX_SKILLS]

    def format_prompt_block(self) -> str:
        if not self.skills:
            return ""
        lines = ["## Available Skills", ""]
        total = 0
        for skill in self.skills:
            block = (
                f"### Skill: {skill['name']}\n"
                f"Description: {skill.get('description', '')}\n"
                f"Invocation: {skill.get('invocation', '')}\n"
                f"Path: {skill.get('path', '')}\n"
            )
            if skill.get("body"):
                block += f"\n{skill['body']}\n"
            block += "\n"
            if total + len(block) > MAX_SKILLS_PROMPT:
                lines.append("(... more skills truncated)")
                break
            lines.append(block)
            total += len(block)
        return "\n".join(lines)

# ---------------------------------------------------------------------------
# 记忆
# ---------------------------------------------------------------------------

class MemoryStore:

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.memory_dir = workspace_dir / "memory" / "daily"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._vector_cache: OrderedDict[tuple[str, str, str, int], list[float]] = OrderedDict()

    def write_memory(self, content: str, category: str = "general") -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.memory_dir / f"{today}.jsonl"
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "content": content,
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return f"Memory saved to {today}.jsonl ({category})"
        except Exception as exc:
            return f"Error writing memory: {exc}"

    def load_evergreen(self) -> str:
        path = self.workspace_dir / "MEMORY.md"
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _load_all_chunks(self) -> list[dict[str, str]]:
        chunks: list[dict[str, str]] = []
        evergreen = self.load_evergreen()
        if evergreen:
            for para in evergreen.split("\n\n"):
                para = para.strip()
                if para:
                    chunks.append({"path": "MEMORY.md", "text": para})
        if self.memory_dir.is_dir():
            for jf in sorted(self.memory_dir.glob("*.jsonl")):
                try:
                    for line in jf.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        text = entry.get("content", "")
                        if text:
                            cat = entry.get("category", "")
                            label = f"{jf.name} [{cat}]" if cat else jf.name
                            chunks.append({"path": label, "text": text})
                except Exception:
                    continue
        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())
        return [t for t in tokens if len(t) > 1 or "\u4e00" <= t <= "\u9fff"]

    @staticmethod
    def _score_chunks_tfidf(query: str, chunks: list[dict[str, str]]) -> list[dict[str, Any]]:
        if not chunks:
            return []
        query_tokens = MemoryStore._tokenize(query)
        if not query_tokens:
            return []

        chunk_tokens = [MemoryStore._tokenize(chunk["text"]) for chunk in chunks]

        df: dict[str, int] = {}
        for tokens in chunk_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        n = len(chunks)

        def tfidf(tokens: list[str]) -> dict[str, float]:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            return {t: c * (math.log((n + 1) / (df.get(t, 0) + 1)) + 1) for t, c in tf.items()}

        def cosine(a: dict[str, float], b: dict[str, float]) -> float:
            common = set(a) & set(b)
            if not common:
                return 0.0
            dot = sum(a[k] * b[k] for k in common)
            na = math.sqrt(sum(v * v for v in a.values()))
            nb = math.sqrt(sum(v * v for v in b.values()))
            return dot / (na * nb) if na and nb else 0.0

        qvec = tfidf(query_tokens)
        scored: list[dict[str, Any]] = []
        for i, tokens in enumerate(chunk_tokens):
            if not tokens:
                continue
            score = cosine(qvec, tfidf(tokens))
            if score > 0.0:
                scored.append({"chunk": chunks[i], "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def search_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        chunks = self._load_all_chunks()
        scored = self._score_chunks_tfidf(query, chunks)
        output: list[dict[str, Any]] = []
        for item in scored[:top_k]:
            snippet = item["chunk"]["text"]
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            output.append({
                "path": item["chunk"]["path"],
                "score": round(item["score"], 4),
                "snippet": snippet,
            })
        return output

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = text.lower().replace("\\", "/")
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[^\w\s/\-.:#@\u4e00-\u9fff]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @staticmethod
    def _char_ngrams(text: str, n: int) -> list[str]:
        compact = re.sub(r"\s+", "", text)
        if len(compact) < n:
            return []
        return [compact[i:i + n] for i in range(len(compact) - n + 1)]

    @staticmethod
    def _stable_digest(feature: str) -> bytes:
        return hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()

    @staticmethod
    def _stable_key(text: str, path: str = "", category: str = "") -> str:
        base = "\n".join([text, path, category])
        return hashlib.blake2b(base.encode("utf-8"), digest_size=16).hexdigest()

    @staticmethod
    def _feature_stream(text: str, path: str = "", category: str = "") -> list[tuple[str, float]]:
        normalized = MemoryStore._normalize_text(text)
        if not normalized:
            return []

        freq: dict[str, int] = {}
        weighted_features: list[tuple[str, float]] = []

        def add_feature(name: str, base_weight: float) -> None:
            freq[name] = freq.get(name, 0) + 1
            weighted_features.append((name, base_weight))

        for token in MemoryStore._tokenize(normalized):
            add_feature(f"tok:{token}", LOCAL_EMBED_TOKEN_WEIGHT)

        for gram in MemoryStore._char_ngrams(normalized, 2):
            add_feature(f"bg:{gram}", LOCAL_EMBED_BIGRAM_WEIGHT)

        for gram in MemoryStore._char_ngrams(normalized, 3):
            add_feature(f"tg:{gram}", LOCAL_EMBED_TRIGRAM_WEIGHT)

        if path:
            normalized_path = MemoryStore._normalize_text(path)
            if normalized_path:
                add_feature(f"path:{normalized_path}", LOCAL_EMBED_PATH_WEIGHT)
                for part in [p for p in normalized_path.split("/") if p]:
                    add_feature(f"pathpart:{part}", LOCAL_EMBED_PATH_PART_WEIGHT)

        if category:
            normalized_category = MemoryStore._normalize_text(category)
            if normalized_category:
                add_feature(f"cat:{normalized_category}", LOCAL_EMBED_CATEGORY_WEIGHT)

        final_features: list[tuple[str, float]] = []
        for name, base_weight in weighted_features:
            tf = freq.get(name, 1)
            final_features.append((name, base_weight / math.sqrt(tf)))
        return final_features

    @staticmethod
    def _hash_vector(text: str, dim: int = LOCAL_EMBED_DIM, *, path: str = "", category: str = "") -> list[float]:
        features = MemoryStore._feature_stream(text, path=path, category=category)
        vec = [0.0] * dim

        projections_per_feature = LOCAL_EMBED_PROJECTIONS
        for feature, weight in features:
            digest = MemoryStore._stable_digest(feature)
            for i in range(projections_per_feature):
                start = (i * 2) % len(digest)
                bucket = int.from_bytes(digest[start:start + 2], "big")
                index = bucket % dim
                sign = 1.0 if ((digest[(start + 5) % len(digest)] >> (i % 8)) & 1) else -1.0
                scale = 0.75 + (digest[(start + 9) % len(digest)] / 255.0) * 0.5
                vec[index] += weight * sign * scale

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _cached_hash_vector(
            self,
            text: str,
            dim: int = LOCAL_EMBED_DIM,
            *,
            path: str = "",
            category: str = "",
    ) -> list[float]:
        cache_key = (self._stable_key(text, path=path, category=category), path, category, dim)
        cached = self._vector_cache.get(cache_key)
        if cached is not None:
            self._vector_cache.move_to_end(cache_key)
            return cached
        vector = self._hash_vector(text, dim=dim, path=path, category=category)
        self._vector_cache[cache_key] = vector
        if len(self._vector_cache) > LOCAL_EMBED_CACHE_MAX:
            self._vector_cache.popitem(last=False)
        return vector

    @staticmethod
    def _vector_cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    @staticmethod
    def _jaccard_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
        set_a, set_b = set(tokens_a), set(tokens_b)
        union = len(set_a | set_b)
        return len(set_a & set_b) / union if union else 0.0

    def _vector_search(self, query: str, chunks: list[dict[str, str]], top_k: int = 10) -> list[dict[str, Any]]:
        q_vec = self._cached_hash_vector(query)
        scored = []
        for chunk in chunks:
            path = chunk.get("path", "")
            category_match = re.search(r"\[(.*?)]", path)
            category = category_match.group(1) if category_match else ""
            score = self._vector_cosine(
                q_vec,
                self._cached_hash_vector(chunk["text"], path=path, category=category),
            )
            if score > 0.0:
                scored.append({"chunk": chunk, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _keyword_search(self, query: str, chunks: list[dict[str, str]], top_k: int = 10) -> list[dict[str, Any]]:
        return self._score_chunks_tfidf(query, chunks)[:top_k]

    @staticmethod
    def _merge_hybrid_results(
            vector_results: list[dict[str, Any]],
            keyword_results: list[dict[str, Any]],
            vector_weight: float = 0.7,
            text_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for result in vector_results:
            key = result["chunk"]["text"][:120]
            merged[key] = {"chunk": result["chunk"], "score": result["score"] * vector_weight}
        for result in keyword_results:
            key = result["chunk"]["text"][:120]
            if key in merged:
                merged[key]["score"] += result["score"] * text_weight
            else:
                merged[key] = {"chunk": result["chunk"], "score": result["score"] * text_weight}
        ranked = list(merged.values())
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    @staticmethod
    def _temporal_decay(results: list[dict[str, Any]], decay_rate: float = 0.01) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        for result in results:
            path = result["chunk"].get("path", "")
            match = re.search(r"(\d{4}-\d{2}-\d{2})", path)
            if not match:
                continue
            try:
                chunk_date = datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            age_days = (now - chunk_date).total_seconds() / 86400.0
            result["score"] *= math.exp(-decay_rate * age_days)
        return results

    @staticmethod
    def _mmr_rerank(results: list[dict[str, Any]], lambda_param: float = 0.7) -> list[dict[str, Any]]:
        if len(results) <= 1:
            return results
        tokenized = [MemoryStore._tokenize(result["chunk"]["text"]) for result in results]
        selected: list[int] = []
        remaining = list(range(len(results)))
        reranked: list[dict[str, Any]] = []
        while remaining:
            best_idx = remaining[0]
            best_score = float("-inf")
            for idx in remaining:
                diversity_penalty = max(
                    (MemoryStore._jaccard_similarity(tokenized[idx], tokenized[sel]) for sel in selected),
                    default=0.0,
                )
                score = lambda_param * results[idx]["score"] - (1.0 - lambda_param) * diversity_penalty
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
            reranked.append(results[best_idx])
        return reranked

    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        chunks = self._load_all_chunks()
        if not chunks:
            return []
        keyword_results = self._keyword_search(query, chunks, top_k=10)
        vector_results = self._vector_search(query, chunks, top_k=10)
        merged = self._merge_hybrid_results(vector_results, keyword_results)
        decayed = self._temporal_decay(merged)
        reranked = self._mmr_rerank(decayed)
        output = []
        for result in reranked[:top_k]:
            snippet = result["chunk"]["text"]
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            output.append({
                "path": result["chunk"]["path"],
                "score": round(result["score"], 4),
                "snippet": snippet,
            })
        return output

    def get_stats(self) -> dict[str, Any]:
        evergreen = self.load_evergreen()
        daily_files = list(self.memory_dir.glob("*.jsonl")) if self.memory_dir.is_dir() else []
        total_entries = 0
        for file in daily_files:
            try:
                total_entries += sum(1 for line in file.read_text(encoding="utf-8").splitlines() if line.strip())
            except Exception:
                continue
        return {
            "evergreen_chars": len(evergreen),
            "daily_files": len(daily_files),
            "daily_entries": total_entries,
        }

# ---------------------------------------------------------------------------
# System Prompt 组装
# ---------------------------------------------------------------------------

def build_system_prompt(
        mode: str = "full",
        bootstrap: dict[str, str] | None = None,
        skills_block: str = "",
        memory_context: str = "",
        agent_id: str = "main",
        channel: str = "cli"
) -> str:
    if bootstrap is None:
        bootstrap = {}
    sections: list[str] = []

    # 第一层 身份 IDENTITY
    identity = bootstrap.get("IDENTITY.md", "").strip()
    sections.append(identity if identity else "You are a helpful personal AI assistant.")

    # 第二层 灵魂 SOUL
    if mode == "full":
        soul = bootstrap.get("SOUL.md", "").strip()
        if soul:
            sections.append(f"## Personality\n\n{soul}")

    # 第三层 工具 TOOL
    tools_md = bootstrap.get("TOOLS.md", "").strip()
    if tools_md:
        sections.append(f"## Tool Usage Guidelines\n\n{tools_md}")

    # 第四层 技能 Skill
    if mode == "full" and skills_block:
        sections.append(skills_block)

    # 第五层 记忆 Memory
    if mode == "full":
        mem_md = bootstrap.get("MEMORY.md", "").strip()
        parts: list[str] = []
        if mem_md:
            parts.append(f"### Evergreen Memory\n\n{mem_md}")
        if memory_context:
            parts.append(f"### Recalled Memories (auto-searched)\n\n{memory_context}")
        if parts:
            sections.append("## Memory\n\n" + "\n\n".join(parts))
        sections.append(
            "## Memory Instructions\n\n"
            "- Use memory_write to save important user facts and preferences.\n"
            "- Reference remembered facts naturally in conversation.\n"
            "- Use memory_search to recall specific past information."
        )

    # 第六层 其余Bootstrap文件
    if mode in ("full", "minimal"):
        for name in ["HEARTBEAT.md", "BOOTSTRAP.md", "AGENTS.md", "USER.md"]:
            content = bootstrap.get(name, "").strip()
            if content:
                sections.append(f"## {name.replace('.md', '')}\n\n{content}")

    # 第七层 Runtime Context
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections.append(
        f"## Runtime Context\n\n"
        f"- Agent ID: {agent_id}\n- Model: {MODEL_ID}\n"
        f"- Channel: {channel}\n- Current time: {now}\n- Prompt mode: {mode}"
    )

    # 第八层 Channel Hint
    hints = {
        "cli": "You are responding via a terminal REPL. Markdown is supported.",
        "feishu": "You are responding via Feishu. Keep messages concise.",
    }
    sections.append(f"## Channel\n\n{hints.get(channel, f'You are responding via {channel}.')}")

    return "\n\n".join(sections)

# ---------------------------------------------------------------------------
# 标准化 Agent ID
# ---------------------------------------------------------------------------

VALID_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
INVALID_CHARS_RE = re.compile(r"[^a-z0-9_-]+")
DEFAULT_AGENT_ID = "main"

def normalize_agent_id(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        return DEFAULT_AGENT_ID
    if VALID_ID_RE.match(trimmed):
        return trimmed.lower()
    cleaned = INVALID_CHARS_RE.sub("-", trimmed.lower().strip("-")[:64])
    return cleaned or DEFAULT_AGENT_ID

# ---------------------------------------------------------------------------
# 五层路由解析（peer_id, guild_id, account_id, channel, default）
# ---------------------------------------------------------------------------

@dataclass
class Binding:
    agent_id: str
    tier: int
    match_key: str # "peer_id" | "guild_id" | "account_id" | "channel" | "default"
    match_value: str
    priority: int = 0 # 大数优先

    def display(self) -> str:
        names = {1: "peer", 2: "guild", 3: "account", 4: "channel", 5: "default"}
        label = names.get(self.tier, f"tier-{self.tier}")
        return f"[{label}] {self.match_key}={self.match_value} -> agent:{self.agent_id} (pri={self.priority})"

class BindingTable:
    def __init__(self) -> None:
        self._bindings: list[Binding] = []

    def add(self, binding: Binding) -> None:
        self._bindings.append(binding)
        self._bindings.sort(key=lambda b: (b.tier, -b.priority))

    def remove(self, agent_id: str, match_key: str, match_value: str) -> bool:
        before = len(self._bindings)
        self._bindings = [
            b for b in self._bindings
            if not (b.agent_id == agent_id and b.match_key == match_key
                    and b.match_value == match_value)
        ]
        return len(self._bindings) < before

    def list_all(self) -> list[Binding]:
        return list(self._bindings)

    # 动态选择代理
    def resolve(self, channel: str = "", account_id: str = "",
                guild_id: str = "", peer_id: str = "") -> tuple[str | None, Binding | None]:
        for b in self._bindings:
            if b.tier == 1 and b.match_key == "peer_id":
                if ":" in b.match_value:
                    if b.match_value == f"{channel}:{peer_id}":
                        return b.agent_id, b
                elif b.match_value == peer_id:
                    return b.agent_id, b
            elif b.tier == 2 and b.match_key == "guild_id" and b.match_value == guild_id:
                return b.agent_id, b
            elif b.tier == 3 and b.match_key == "account_id" and b.match_value == account_id:
                return b.agent_id, b
            elif b.tier == 4 and b.match_key == "channel" and b.match_value == channel:
                return b.agent_id, b
            elif b.tier == 5 and b.match_key == "default":
                return b.agent_id, b
        return None, None

# ---------------------------------------------------------------------------
# 构建会话键 - dm_scope 控制隔离策略
# ---------------------------------------------------------------------------

def build_session_key(agent_id: str, channel: str = "", account_id: str = "",
                      peer_id: str = "", dm_scope: str = "per-peer") -> str:
    aid = normalize_agent_id(agent_id)
    ch = (channel or "unknown").strip().lower()
    acc = (account_id or "default").strip().lower()
    pid = (peer_id or "").strip().lower()
    if dm_scope == "per-account-channel-peer" and pid:
        return f"agent:{aid}:{ch}:{acc}:direct:{pid}"
    if dm_scope == "per-channel-peer" and pid:
        return f"agent:{aid}:{ch}:direct:{pid}"
    if dm_scope == "per-peer" and pid:
        return f"agent:{aid}:direct:{pid}"
    return f"agent:{aid}:main"

# ---------------------------------------------------------------------------
# Agent 配置和管理
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    id: str
    name: str
    personality: str = ""
    model: str = ""
    dm_scope: str = "per-peer"

    @property
    def effective_model(self) -> str:
        return self.model or MODEL_ID

    def system_prompt(self) -> str:
        parts = [f"You are {self.name}."]
        if self.personality:
            parts.append(f"Your personality: {self.personality}")
        parts.append("Answer questions helpfully and stay in character.")
        return " ".join(parts)

class AgentManager:
    def __init__(self, agents_base: Path | None = None) -> None:
        self._agents: dict[str, AgentConfig] = {}
        self._agents_base = agents_base or AGENTS_DIR
        self._sessions: dict[str, list[dict]] = {}

    def register(self, config: AgentConfig) -> None:
        aid = normalize_agent_id(config.id)
        config.id = aid
        self._agents[aid] = config
        agent_dir = self._agents_base / aid
        (agent_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (WORKDIR / f"workspace-{aid}").mkdir(parents=True, exist_ok=True)

    def get_agent(self, agent_id: str) -> AgentConfig | None:
        return self._agents.get(normalize_agent_id(agent_id))

    def list_agents(self) -> list[AgentConfig]:
        return list(self._agents.values())

    def get_session(self, session_key: str) -> list[dict]:
        if session_key not in self._sessions:
            self._sessions[session_key] = []
        return self._sessions[session_key]

    def list_sessions(self, agent_id: str = "") -> dict[str, int]:
        aid = normalize_agent_id(agent_id) if agent_id else ""
        return {k: len(v) for k, v in self._sessions.items()
                if not aid or k.startswith(f"agent:{aid}:")}

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
# LLM Adapter
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    @abstractmethod
    def chat(
            self,
            model: str,
            system: str,
            messages: list[dict],
            max_tokens: int = 8096,
            tools: list[dict] | None = None,
             ) -> dict:
        pass

# Anthropic Adapter
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, base_url: str | None = None):
        if not HAS_ANTHROPIC:
            raise RuntimeError("Please install anthropic: pip install anthropic")
        self.client = Anthropic(api_key=api_key, base_url=base_url)

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        converted = []
        for msg in messages:
            role = msg["role"]
            if role not in ("user", "assistant"):
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                converted.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}]
                })
            elif isinstance(content, list):
                anthropic_content = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            anthropic_content.append(block)
                        elif block.get("type") == "tool_result":
                            anthropic_content.append(block)
                        elif block.get("type") == "text":
                            anthropic_content.append(block)
                        else:
                            anthropic_content.append({"type": "text", "text": str(block)})
                    else:
                        anthropic_content.append({"type": "text", "text": str(block)})
                converted.append({
                    "role": role,
                    "content": anthropic_content or [{"type": "text", "text": ""}],
                })
        return converted

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
        return anthropic_tools

    def chat(self, model: str, system: str, messages: list[dict], max_tokens: int = 8096, tools: list[dict] | None = None) -> dict:
        converted_msgs = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        resp = self.client.messages.create(
            model=model,
            system=system,
            messages=converted_msgs,
            max_tokens=max_tokens,
            tools=anthropic_tools,
        )

        text_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })

        finish_reason = resp.stop_reason
        if finish_reason == "end_turn":
            finish_reason = "stop"
        elif finish_reason == "tool_use":
            finish_reason = "tool_calls"

        return {
            "text": "\n".join(text_parts),
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "raw": resp
        }

# OpenAI Adapter
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, base_url: str | None = None):
        if not HAS_OPENAI:
            raise RuntimeError("Please install openai: pip install openai")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, model: str, system: str, messages: list[dict], max_tokens: int = 8096, tools: list[dict] | None = None) -> dict:
        full_messages = [{"role": "system", "content": system}] + messages

        resp = self.client.chat.completions.create(
            model=model,
            messages=full_messages,
            max_tokens=max_tokens,
            tools=tools,
        )

        msg = resp.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })

        return {
            "text": msg.content or "",
            "tool_calls": tool_calls,
            "finish_reason": resp.choices[0].finish_reason,
            "raw": resp
        }

# DashScope Adapter
class DashScopeClient(LLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def chat(self, model: str, system: str, messages: list[dict], max_tokens: int = 8096, tools: list[dict] | None = None) -> dict:
        full_messages = [{"role": "system", "content": system}] + messages

        resp = dashscope.MultiModalConversation.call(
            api_key=self.api_key,
            model=model,
            messages=full_messages,
            max_tokens=max_tokens,
            result_format="message",
            tools=tools,
        )

        if resp.status_code != 200:
            raise Exception(f"DashScope error {resp.status_code}: {resp.message}")

        choice = resp.output.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason

        text = ""
        if hasattr(msg, "content"):
            if isinstance(msg.content, list):
                text = "\n".join(x.get("text", "") for x in msg.content if isinstance(x, dict))
            else:
                text = str(msg.content)

        tool_calls = []
        if finish_reason == "tool_calls" and hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"].get("arguments", "{}")
                    }
                })

        return {
            "text": text,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "raw": resp
        }

# LLM 工厂
def create_llm_client() -> LLMClient:
    provider = LLM_PROVIDER
    if not provider:
        if ANTHROPIC_API_KEY:
            provider = "anthropic"
        elif OPENAI_API_KEY:
            provider = "openai"
        elif DASHSCOPE_API_KEY:
            provider = "dashscope"
        else:
            raise RuntimeError("No LLM API key found. Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, or DASHSCOPE_API_KEY.")

    if provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic provider.")
        return AnthropicClient(ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL)
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider.")
        return OpenAIClient(OPENAI_API_KEY, OPENAI_BASE_URL)
    elif provider == "dashscope":
        if not DASHSCOPE_API_KEY:
            raise RuntimeError("DASHSCOPE_API_KEY is required for DashScope provider.")
        return DashScopeClient(DASHSCOPE_API_KEY)
    else:
        raise RuntimeError(f"Unknown LLM provider: {provider}")

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
    def compact_history(messages: list[dict], client: LLMClient, model: str) -> list[dict]:
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
                    current_messages = self.compact_history(current_messages, client, model)

        raise RuntimeError("guard_api_call: exhausted retries")

# 截断过长文本 TODO:统一truncate工具
def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} total chars]"

# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------
memory_store = MemoryStore(WORKSPACE_DIR)

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
def tool_memory_write(content: str, category: str = "general") -> str:
    print_tool("memory_write", f"[{category}] {content[:60]}...")
    return memory_store.write_memory(content, category)

# 记忆搜索工具
def tool_memory_search(query: str, top_k: int = 5) -> str:
    print_tool("memory_search", query)
    results = memory_store.hybrid_search(query, top_k)
    if not results:
        return "No relevant memories found."
    return "\n".join(f"[{r['path']}] (score: {r['score']}) {r['snippet']}" for r in results)

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
            "description": "Search through saved memory notes.",
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
# 共享事件循环 (持久化后台线程)
# ---------------------------------------------------------------------------

BOOTSTRAP_DATA: dict[str, str] = {}
SKILLS_MANAGER = SkillsManager(WORKSPACE_DIR)
SKILLS_BLOCK = ""

def refresh_intelligence(mode: str = "full") -> None:
    global BOOTSTRAP_DATA, SKILLS_BLOCK
    loader = BootstrapLoader(WORKSPACE_DIR)
    BOOTSTRAP_DATA = loader.load_all(mode=mode)
    SKILLS_MANAGER.discover()
    SKILLS_BLOCK = SKILLS_MANAGER.format_prompt_block()

def format_memory_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    return "\n".join(f"- [{r['path']}] (score: {r['score']}) {r['snippet']}" for r in results)

def auto_recall(user_message: str, top_k: int = 3) -> str:
    return format_memory_results(memory_store.hybrid_search(user_message, top_k=top_k))

def compose_runtime_system_prompt(
        agent_id: str,
        channel: str,
        memory_context: str = "",
        mode: str = "full",
) -> str:
    return build_system_prompt(
        mode=mode,
        bootstrap=BOOTSTRAP_DATA,
        skills_block=SKILLS_BLOCK,
        memory_context=memory_context,
        agent_id=agent_id,
        channel=channel,
    )

_event_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None

def get_event_loop() -> asyncio.AbstractEventLoop:
    global _event_loop, _loop_thread
    if _event_loop is not None and _event_loop.is_running():
        return _event_loop
    _event_loop = asyncio.new_event_loop()
    def _run():
        asyncio.set_event_loop(_event_loop)
        _event_loop.run_forever()
    _loop_thread = threading.Thread(target=_run, daemon=True)
    _loop_thread.start()
    return _event_loop

def run_async(coro: Any) -> Any:
    loop = get_event_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result()

# ---------------------------------------------------------------------------
# 路由解析
# ---------------------------------------------------------------------------

def resolve_route(bindings: BindingTable, mgr: AgentManager,
                  channel: str, peer_id: str,
                  account_id: str = "", guild_id: str = "") -> tuple[str, str]:
    agent_id, matched = bindings.resolve(
        channel=channel, account_id=account_id,
        guild_id=guild_id, peer_id=peer_id,
    )
    if not agent_id:
        agent_id = DEFAULT_AGENT_ID
        print(f"{DIM}[route] No binding matched, default: {agent_id}{RESET}")
    elif matched:
        print(f"{DIM}[route] Matched: {matched.display()}{RESET}")
    agent = mgr.get_agent(agent_id)
    dm_scope = agent.dm_scope if agent else "per-peer"
    sk = build_session_key(agent_id, channel=channel, account_id=account_id,
                           peer_id=peer_id, dm_scope=dm_scope)
    return agent_id, sk

# ---------------------------------------------------------------------------
# Agent 运行器 - 限制并发请求数量
# ---------------------------------------------------------------------------

_agent_semaphore: asyncio.Semaphore | None = None

async def run_agent(mgr: AgentManager, agent_id: str, session_key: str,
                    user_text: str, on_typing: Any = None,
                    channel: str = "unknown",
                    llm_client: LLMClient | None = None) -> str:
    global _agent_semaphore
    if _agent_semaphore is None:
        _agent_semaphore = asyncio.Semaphore(4)
    if llm_client is None:
        llm_client = create_llm_client()
    agent = mgr.get_agent(agent_id)
    if not agent:
        return f"Error: agent '{agent_id}' not found"
    messages = mgr.get_session(session_key)
    messages.append({"role": "user", "content": user_text})
    memory_context = auto_recall(user_text)
    system_prompt = compose_runtime_system_prompt(
        agent_id=agent.id,
        channel=channel,
        memory_context=memory_context,
    )
    async with _agent_semaphore:
        if on_typing:
            on_typing(agent_id, True)
        try:
            return await _agent_loop(llm_client, agent.effective_model, system_prompt, messages)
        finally:
            if on_typing:
                on_typing(agent_id, False)

async def _agent_loop(client: LLMClient, model: str, system: str, messages: list[dict]) -> str:
    for _ in range(15):
        try:
            response = await asyncio.to_thread(
                client.chat,
                model=model,
                system=system,
                messages=messages,
                max_tokens=4096,
                tools=TOOLS,
            )
        except Exception as exc:
            while messages and messages[-1]["role"] != "user":
                messages.pop()
            if messages:
                messages.pop()
            return f"API Error: {exc}"

        finish_reason = response["finish_reason"]
        assistant_text = response["text"]
        tool_calls = response["tool_calls"]

        assistant_msg = {"role": "assistant", "content": assistant_text}
        if tool_calls:
            if isinstance(client, AnthropicClient):
                content_blocks = []
                if assistant_text:
                    content_blocks.append({"type": "text", "text": assistant_text})
                for tc in tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"].get("arguments", "{}")),
                    })
                assistant_msg["content"] = content_blocks
            else:
                assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if finish_reason == "stop":
            return assistant_text or "[no text]"

        if finish_reason == "tool_calls":
            if isinstance(client, AnthropicClient):
                tool_results = []
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"].get("arguments", "{}"))
                    tool_call_id = tc.get("id")
                    result = process_tool_call(tool_name, tool_args)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})
            else:
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"].get("arguments", "{}"))
                    tool_call_id = tc.get("id")
                    result = process_tool_call(tool_name, tool_args)
                    messages.append({"role": "tool", "content": result, "tool_call_id": tool_call_id})
            continue

        return assistant_text or f"[finish_reason={finish_reason}]"
    return "[max iterations reached]"

# ---------------------------------------------------------------------------
# Gateway 服务器 (WebSocket, JSON-RPC 2.0)
# ---------------------------------------------------------------------------

class GatewayServer:
    def __init__(self, mgr: AgentManager, bindings: BindingTable,
                 llm_client: LLMClient | None = None,
                 host: str = "localhost", port: int = 8765) -> None:
        self._mgr = mgr
        self._bindings = bindings
        self._llm_client = llm_client or create_llm_client()
        self._host, self._port = host, port
        self._clients: set[Any] = set()
        self._start_time = time.monotonic()
        self._server: Any = None
        self._running = False

    async def start(self) -> None:
        try:
            import websockets
        except ImportError:
            print(f"{RED}websockets not installed. pip install websockets{RESET}")
        self._start_time = time.monotonic()
        self._running = True
        self._server = await websockets.serve(self._handle, self._host, self._port)
        print(f"{GREEN}Gateway started ws://{self._host}:{self._port}{RESET}")

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._running = False

    async def _handle(self, ws: Any, path: str = "") -> None:
        self._clients.add(ws)
        try:
            async for raw in ws:
                resp = await self._dispatch(raw)
                if resp:
                    await ws.send(json.dumps(resp))
        except Exception:
            pass
        finally:
            self._clients.discard(ws)

    def _typing_cb(self, agent_id: str, typing: bool) -> None:
        msg = json.dumps({"jsonrpc": "2.0", "method": "typing",
                          "params": {"agent_id": agent_id, "typing": typing}})
        for ws in list(self._clients):
            try:
                asyncio.ensure_future(ws.send(msg))
            except Exception:
                self._clients.discard(ws)

    async def _dispatch(self, raw: str) -> dict | None:
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            return {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}
        rid, method, params = req.get("id"), req.get("method", ""), req.get("params", {})
        methods = {
            "send": self._m_send, "bindings.set": self._m_bind_set,
            "bindings.list": self._m_bind_list, "sessions.list": self._m_sessions,
            "agents.list": self._m_agents, "status": self._m_status,
        }
        handler = methods.get(method)
        if not handler:
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Unknown: {method}"}, "id": rid}
        try:
            return {"jsonrpc": "2.0", "result": await handler(params), "id": rid}
        except Exception as exc:
            return {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(exc)}, "id": rid}

    async def _m_send(self, p: dict) -> dict:
        text = p.get("text", "")
        if not text:
            raise ValueError("text is required")
        ch = p.get("channel", "websocket")
        pid = p.get("peer_id", "ws-client")
        acc = p.get("account_id", "")
        gid = p.get("guild_id", "")
        if p.get("agent_id"):
            aid = normalize_agent_id(p["agent_id"])
            a = self._mgr.get_agent(aid)
            sk = build_session_key(aid, channel=ch, account_id=acc, peer_id=pid,
                                   dm_scope=a.dm_scope if a else "per-peer")
        else:
            aid, sk = resolve_route(
                self._bindings,
                self._mgr,
                channel=ch,
                peer_id=pid,
                account_id=acc,
                guild_id=gid,
            )
        reply = await run_agent(
            self._mgr,
            aid,
            sk,
            text,
            on_typing=self._typing_cb,
            channel=ch,
            llm_client=self._llm_client,
        )
        return {"agent_id": aid, "session_key": sk, "reply": reply}

    async def _m_bind_set(self, p: dict) -> dict:
        b = Binding(agent_id=normalize_agent_id(p.get("agent_id", "")),
                    tier=int(p.get("tier", 5)), match_key=p.get("match_key", "default"),
                    match_value=p.get("match_value", "*"), priority=int(p.get("priority", 0)))
        self._bindings.add(b)
        return {"ok": True, "binding": b.display()}

    async def _m_bind_list(self, p: dict) -> list[dict]:
        return [{"agent_id": b.agent_id, "tier": b.tier, "match_key": b.match_key,
                 "match_value": b.match_value, "priority": b.priority}
                for b in self._bindings.list_all()]

    async def _m_sessions(self, p: dict) -> dict:
        return self._mgr.list_sessions(p.get("agent_id", ""))

    async def _m_agents(self, p: dict) -> list[dict]:
        return [{"id": a.id, "name": a.name, "model": a.effective_model,
                 "dm_scope": a.dm_scope, "personality": a.personality}
                for a in self._mgr.list_agents()]

    async def _m_status(self, p: dict) -> dict:
        return {"running": self._running,
                "uptime_seconds": round(time.monotonic() - self._start_time, 1),
                "connected_clients": len(self._clients),
                "agent_count": len(self._mgr.list_agents()),
                "binding_count": len(self._bindings.list_all())}

# ---------------------------------------------------------------------------
# 默认 Agent 初始化
# ---------------------------------------------------------------------------

def setup_default_agent(agent_mgr: AgentManager, bindings: BindingTable) -> None:
    """初始化默认 Agent"""
    agent_mgr.register(AgentConfig(
        id="main",
        name="Main Agent",
        personality="",
    ))
    bindings.add(Binding(
        agent_id="main",
        tier=5,
        match_key="default",
        match_value="*",
    ))

# ---------------------------------------------------------------------------
# Read-Eval-Print Loop
# ---------------------------------------------------------------------------

def handle_repl_command(
        command: str,
        store: SessionStore,
        guard: ContextGuard,
        messages: list[dict],
        mgr: ChannelManager,
        llm_client: LLMClient,
        model_id: str,
        bindings: BindingTable | None = None,
        agent_mgr: AgentManager | None = None,
        force_agent_id: Optional[str] = None,
        set_force_agent: Optional[Callable[..., Any]] = None,
        gw_server: Optional["GatewayServer"] = None,
        set_gw_server: Optional[Callable[..., Any]] = None,
        bootstrap_data: dict[str, str] | None = None,
        skills_mgr: SkillsManager | None = None,
        skills_block: str = "",
) -> tuple[bool, list[dict], Optional[str], Optional["GatewayServer"]]:
    parts = command.strip().split(maxsplit = 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/new":
        label = arg or ""
        sid = store.create_session(label)
        print_session(f"Created new session: {sid}" + (f" ({label})" if label else ""))
        return True, [], force_agent_id, gw_server
    elif cmd == "/list":
        sessions = store.list_sessions()
        if not sessions:
            print_info("No sessions found.")
            return True, messages, force_agent_id, gw_server

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
        return True, messages, force_agent_id, gw_server

    elif cmd == "/switch":
        if not arg:
            print_warn("Usage: /switch <session_id>")
            return True, messages, force_agent_id, gw_server
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
            return True, messages, force_agent_id, gw_server
        if len(matched) > 1:
            print_warn(f"Ambiguous prefix, matches: {', '.join(matched)}")
            return True, messages, force_agent_id, gw_server

        new_sid = matched[0]
        new_messages = store.load_session(new_sid)
        print_session(f" Switched to session: {new_sid} ({len(new_messages)} messages)")
        return True, new_messages, force_agent_id, gw_server

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
        return True, messages, force_agent_id, gw_server

    elif cmd == "/compact":
        if len(messages) <= 2:
            print_info("Too few messages to compact (need > 2).")
            return True, messages, force_agent_id, gw_server
        print_session("Compacting history...")
        new_messages = guard.compact_history(messages, llm_client, model_id)
        print_session(f"{len(messages)} -> {len(new_messages)} messages")
        return True, new_messages, force_agent_id, gw_server

    elif cmd == "/channels":
        channels = mgr.list_channels()
        if channels:
            print_channel("Channels:")
            for name in channels:
                print_channel(f"  - {name}")
        else:
            print_info("No channels.")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/soul":
        print_section("SOUL.md")
        soul = (bootstrap_data or {}).get("SOUL.md", "")
        print(soul if soul else f"{DIM}(SOUL.md not found in {WORKSPACE_DIR}){RESET}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/skills":
        mgr_skills = skills_mgr or SKILLS_MANAGER
        print_section("Skills")
        if not mgr_skills.skills:
            print(f"{DIM}(no skills discovered){RESET}")
        else:
            for skill in mgr_skills.skills:
                print(f"- {skill['name']}: {skill.get('description', '')}")
                print(f"  path={skill.get('path', '')}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/memory":
        print_section("Memory")
        stats = memory_store.get_stats()
        print(f"  MEMORY.md chars: {stats['evergreen_chars']}")
        print(f"  daily files: {stats['daily_files']}")
        print(f"  daily entries: {stats['daily_entries']}")
        print(f"  dir: {memory_store.memory_dir}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/search":
        if not arg:
            print_warn("Usage: /search <query>")
            return True, messages, force_agent_id, gw_server
        print_section(f"Memory Search: {arg}")
        results = memory_store.hybrid_search(arg)
        if not results:
            print(f"{DIM}(no relevant memories){RESET}")
        else:
            print(format_memory_results(results))
        return True, messages, force_agent_id, gw_server

    elif cmd == "/prompt":
        prompt = build_system_prompt(
            mode="full",
            bootstrap=bootstrap_data or BOOTSTRAP_DATA,
            skills_block=skills_block or SKILLS_BLOCK,
            memory_context=auto_recall("show prompt"),
            agent_id=force_agent_id or DEFAULT_AGENT_ID,
            channel="cli",
        )
        print_section("System Prompt")
        if len(prompt) > 3000:
            print(prompt[:3000])
            print(f"\n{DIM}... ({len(prompt) - 3000} more chars, total {len(prompt)}){RESET}")
        else:
            print(prompt)
        return True, messages, force_agent_id, gw_server

    elif cmd == "/bootstrap":
        print_section("Bootstrap")
        data = bootstrap_data or BOOTSTRAP_DATA
        if not data:
            print(f"{DIM}(no bootstrap files loaded from {WORKSPACE_DIR}){RESET}")
        else:
            for name, content in data.items():
                print(f"- {name}: {len(content)} chars")
            print(f"  total: {sum(len(v) for v in data.values())} chars")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/reload_intelligence":
        refresh_intelligence()
        print_info(
            f"Reloaded intelligence: bootstrap={len(BOOTSTRAP_DATA)} files, "
            f"skills={len(SKILLS_MANAGER.skills)}"
        )
        return True, messages, force_agent_id, gw_server

    elif cmd == "/accounts":
        accounts = mgr.accounts
        if accounts:
            print_channel("Accounts:")
            for acc in accounts:
                masked = acc.token[:8] + "..." if len(acc.token) > 8 else "(none)"
                print_channel(f"- {acc.channel}/{acc.account_id}  token={masked}")
        else:
            print_info("No accounts.")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/bindings":
        if bindings is None:
            print_warn("Bindings not available")
            return True, messages, force_agent_id, gw_server
        all_b = bindings.list_all()
        if not all_b:
            print_info("(no bindings)")
        else:
            print(f"\n{BOLD}Route Bindings ({len(all_b)}):{RESET}")
            for b in all_b:
                c = [MAGENTA, BLUE, CYAN, GREEN, DIM][min(b.tier - 1, 4)]
                print(f"  {c}{b.display()}{RESET}")
            print()
        return True, messages, force_agent_id, gw_server

    elif cmd == "/route":
        if bindings is None or agent_mgr is None:
            print_warn("Bindings or AgentManager not available")
            return True, messages, force_agent_id, gw_server
        parts = arg.strip().split()
        if len(parts) < 2:
            print_warn("Usage: /route <channel> <peer_id> [account_id] [guild_id]")
            return True, messages, force_agent_id, gw_server
        ch, pid = parts[0], parts[1]
        acc = parts[2] if len(parts) > 2 else ""
        gid = parts[3] if len(parts) > 3 else ""
        aid, sk = resolve_route(bindings, agent_mgr, channel=ch, peer_id=pid, account_id=acc, guild_id=gid)
        a = agent_mgr.get_agent(aid)
        print(f"\n{BOLD}Route Resolution:{RESET}")
        print(f"  {DIM}Input:   ch={ch} peer={pid} acc={acc or '-'} guild={gid or '-'}{RESET}")
        print(f"  {CYAN}Agent:   {aid} ({a.name if a else '?'}){RESET}")
        print(f"  {GREEN}Session: {sk}{RESET}\n")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/agents":
        if agent_mgr is None:
            print_warn("AgentManager not available")
            return True, messages, force_agent_id, gw_server
        agents = agent_mgr.list_agents()
        if not agents:
            print_info("(no agents)")
        else:
            print(f"\n{BOLD}Agents ({len(agents)}):{RESET}")
            for a in agents:
                print(f"  {CYAN}{a.id}{RESET} ({a.name})  model={a.effective_model}  dm_scope={a.dm_scope}")
                if a.personality:
                    print(f"    {DIM}{a.personality[:70]}{'...' if len(a.personality) > 70 else ''}{RESET}")
            print()
        return True, messages, force_agent_id, gw_server

    elif cmd == "/switch_agent":
        if set_force_agent is None:
            print_warn("Agent switching not available")
            return True, messages, force_agent_id, gw_server
        if not arg:
            print_info(f"force_agent: {force_agent_id or '(off)'}")
            return True, messages, force_agent_id, gw_server
        if arg.lower() == "off":
            set_force_agent(None)
            force_agent_id = None
            print_info("Routing mode restored.")
        else:
            if agent_mgr is not None and agent_mgr.get_agent(arg):
                force_agent_id = normalize_agent_id(arg)
                set_force_agent(force_agent_id)
                print_info(f"Forcing agent: {force_agent_id}")
            else:
                print_warn(f"Agent not found: {arg}")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/gateway":
        if set_gw_server is None:
            print_warn("Gateway not available")
            return True, messages, force_agent_id, gw_server
        if arg == "start":
            if gw_server is None:
                if agent_mgr is None or bindings is None:
                    print_warn("AgentManager or Bindings not available")
                    return True, messages, force_agent_id, gw_server
                new_gw = GatewayServer(agent_mgr, bindings)
                asyncio.run_coroutine_threadsafe(new_gw.start(), get_event_loop()).result()
                set_gw_server(new_gw)
                gw_server = new_gw
                print_info("Gateway started: ws://localhost:8765")
            else:
                print_info("Gateway already running")
        elif arg == "stop":
            if gw_server is not None:
                asyncio.run_coroutine_threadsafe(gw_server.stop(), get_event_loop()).result()
                set_gw_server(None)
                gw_server = None
                print_info("Gateway stopped")
            else:
                print_info("Gateway not running")
        elif arg == "status":
            if gw_server:
                print_info(f"Gateway running: ws://localhost:8765")
                print_info(f"  Agents: {len(agent_mgr.list_agents()) if agent_mgr else 0}")
                print_info(f"  Bindings: {len(bindings.list_all()) if bindings else 0}")
            else:
                print_info("Gateway not running")
        else:
            print_info("Usage: /gateway [start|stop|status]")
        return True, messages, force_agent_id, gw_server

    elif cmd == "/sessions":
        if agent_mgr is None:
            print_warn("AgentManager not available")
            return True, messages, force_agent_id, gw_server
        s = agent_mgr.list_sessions()
        if not s:
            print_info("(no sessions)")
        else:
            print(f"\n{BOLD}Sessions ({len(s)}):{RESET}")
            for k, n in sorted(s.items()):
                print(f"  {GREEN}{k}{RESET} ({n} msgs)")
            print()
        return True, messages, force_agent_id, gw_server

    elif cmd == "/help":
        print_info("  Session Commands:")
        print_info("    /new [label]       Create a new session")
        print_info("    /list              List all sessions")
        print_info("    /switch <id>       Switch to a session (prefix match)")
        print_info("    /context           Show context token usage")
        print_info("    /compact           Manually compact conversation history")
        print_info("    /soul              Show loaded SOUL.md")
        print_info("    /skills            List discovered skills")
        print_info("    /memory            Show memory stats")
        print_info("    /search <query>    Search long-term memory")
        print_info("    /prompt            Preview composed system prompt")
        print_info("    /bootstrap         Show loaded bootstrap files")
        print_info("    /reload_intelligence Reload bootstrap files and skills")
        print_info("")
        print_info("  Gateway Commands:")
        print_info("    /bindings          List all route bindings")
        print_info("    /route <ch> <peer> Test route resolution")
        print_info("    /agents            List all agents")
        print_info("    /switch_agent <id> Force use specific agent")
        print_info("    /switch_agent off  Restore normal routing")
        print_info("    /gateway start     Start WebSocket gateway")
        print_info("    /gateway stop     Stop WebSocket gateway")
        print_info("    /gateway status   Show gateway status")
        print_info("    /sessions         List current sessions")
        print_info("")
        print_info("  Other Commands:")
        print_info("    /channels          List all channels")
        print_info("    /accounts          List configured accounts")
        print_info("    /help              Show this help")
        print_info("    quit / exit        Exit the REPL")
        return True, messages, force_agent_id, gw_server

    return False, messages, force_agent_id, gw_server

# ---------------------------------------------------------------------------
# Agent 交互回合
# ---------------------------------------------------------------------------

def run_agent_turn(
        inbound: InboundMessage,
        conversations: dict[str, list[dict]],
        mgr: ChannelManager,
        client: LLMClient,
        store: SessionStore | None = None,
        model_id: str = MODEL_ID,
        system_prompt: str = SYSTEM_PROMPT,
        session_key: str | None = None,
) -> None:
    sk = session_key or build_session_key(
        DEFAULT_AGENT_ID,
        channel=inbound.channel,
        account_id=inbound.account_id,
        peer_id=inbound.peer_id,
    )
    if sk not in conversations:
        conversations[sk] = []
    messages = conversations[sk]

    should_persisit = (store is not None and inbound.channel == "cli")

    # --- 添加聊天记录到历史 ---
    user_message = {
        "role": "user",
        "content": inbound.text,
    }
    messages.append(user_message)
    if should_persisit:
        user_record = user_message.copy()
        store.append_transcript(store.current_session_id, user_record)

    guard = ContextGuard()

    while True:
        try:
            response = guard.guard_api_call(
                client=client,
                model=model_id,
                max_tokens=8096,
                system=system_prompt,
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

        finish_reason = response["finish_reason"]
        assistant_content = response["text"]
        tool_calls = response["tool_calls"]

        assistant_msg = {
            "role": "assistant",
            "content": assistant_content
        }

        if tool_calls:
            if isinstance(client, AnthropicClient):
                content_blocks = []
                if assistant_content:
                    content_blocks.append({"type": "text", "text": assistant_content})
                for tc in tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]),
                    })
                assistant_msg["content"] = content_blocks
            else:
                assistant_msg["tool_calls"] = tool_calls

        messages.append(assistant_msg)
        if should_persisit:
            store.append_transcript(store.current_session_id, assistant_msg.copy())

        # --- 调用终止条件stop_reason ---
        if finish_reason == "stop":
            if assistant_content:
                ch = mgr.get(inbound.channel)
                if ch:
                    ch.send(inbound.peer_id, assistant_content)
                else:
                    print_assistant(assistant_content)
            break

        elif finish_reason == "tool_calls":
            if isinstance(client, AnthropicClient):
                tool_results = []
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"]["arguments"])
                    tool_call_id = tc["id"]

                    result = process_tool_call(tool_name, tool_args)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": result
                    })

                tool_msg = {
                    "role": "user",
                    "content": tool_results
                }
                messages.append(tool_msg)
                if should_persisit:
                    store.append_transcript(store.current_session_id, tool_msg.copy())
            else:
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"]["arguments"])
                    tool_call_id = tc["id"]

                    result = process_tool_call(tool_name, tool_args)
                    tool_msg = {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call_id
                    }
                    messages.append(tool_msg)

                    if should_persisit:
                        store.append_transcript(store.current_session_id, tool_msg.copy())
            continue

        else:
            print_info(f"[finish_reason]={finish_reason}")
            if assistant_content:
                ch = mgr.get(inbound.channel)
                if ch:
                    ch.send(inbound.peer_id, assistant_content)
                else:
                    print_assistant(assistant_content)
            break

# ---------------------------------------------------------------------------
# 核心: Agent 循环
# ---------------------------------------------------------------------------

def agent_loop() -> None:
    try:
        llm_client = create_llm_client()
    except Exception as e:
        print(f"{RED}Failed to create LLM client: {e}{RESET}")
        sys.exit(1)

    mgr = ChannelManager()
    bindings = BindingTable()
    agent_mgr = AgentManager()

    refresh_intelligence()
    setup_default_agent(agent_mgr, bindings)

    cli = CLIChannel()
    mgr.register(cli)

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

    store = SessionStore(agent_id="MyClaw")
    guard = ContextGuard()

    conversations: dict[str, list[dict]] = {}

    cli_agent = agent_mgr.get_agent(DEFAULT_AGENT_ID)
    cli_sk = build_session_key(
        DEFAULT_AGENT_ID,
        channel="cli",
        account_id="cli-local",
        peer_id="cli-user",
        dm_scope=cli_agent.dm_scope if cli_agent else "per-peer",
    )
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

    force_agent_id: str | None = None
    gw_server: GatewayServer | None = None

    def set_force_agent(aid: str | None) -> None:
        nonlocal force_agent_id
        force_agent_id = aid

    def set_gw_server(gw: GatewayServer | None) -> None:
        nonlocal gw_server
        gw_server = gw

    def cli_reader():
        while not stop_event.is_set():
            msg = cli.receive()
            if msg is None:
                continue
            msg_queue.put(msg)

    print_info("=" * 60)
    print_info(f" Provider: {type(llm_client).__name__}")
    print_info(f" Model: {MODEL_ID}")
    print_info(f" Session: {store.current_session_id}")
    print_info(f" Channels: {', '.join(mgr.list_channels())}")
    print_info(f" Agents: {', '.join(a.id for a in agent_mgr.list_agents())}")
    print_info(f" Bindings: {len(bindings.list_all())}")
    print_info(f" Intelligence workspace: {WORKSPACE_DIR}")
    print_info(f" Bootstrap files: {len(BOOTSTRAP_DATA)}")
    print_info(f" Skills: {len(SKILLS_MANAGER.skills)}")
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
                msg = msg_queue.get(timeout=0.5)
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
                    handled, new_messages, new_force, new_gw = handle_repl_command(
                        msg.text, store, guard, current_messages, mgr,
                        llm_client, MODEL_ID,
                        bindings=bindings, agent_mgr=agent_mgr,
                        force_agent_id=force_agent_id,
                        set_force_agent=set_force_agent,
                        gw_server=gw_server,
                        set_gw_server=set_gw_server,
                        bootstrap_data=BOOTSTRAP_DATA,
                        skills_mgr=SKILLS_MANAGER,
                        skills_block=SKILLS_BLOCK,
                    )
                    conversations[cli_sk] = new_messages
                    if new_force is not None:
                        force_agent_id = new_force
                    if new_gw is not None:
                        gw_server = new_gw
                    if handled:
                        cli.allow_input()
                        continue

            if msg.channel == "cli":
                if force_agent_id:
                    aid = force_agent_id
                    a = agent_mgr.get_agent(aid)
                    if a:
                        sk = build_session_key(aid, channel=msg.channel, account_id=msg.account_id,
                                               peer_id=msg.peer_id, dm_scope=a.dm_scope)
                        result = run_async(run_agent(
                            agent_mgr, aid, sk, msg.text,
                            channel=msg.channel,
                            llm_client=llm_client,
                        ))
                        print_assistant(result)
                    else:
                        print_warn(f"Agent '{aid}' not found")
                else:
                    cli_agent = agent_mgr.get_agent(DEFAULT_AGENT_ID)
                    memory_context = auto_recall(msg.text)
                    system_prompt = compose_runtime_system_prompt(
                        agent_id=DEFAULT_AGENT_ID,
                        channel=msg.channel,
                        memory_context=memory_context,
                    )
                    run_agent_turn(
                        msg,
                        conversations,
                        mgr,
                        llm_client,
                        store=store,
                        model_id=cli_agent.effective_model if cli_agent else MODEL_ID,
                        system_prompt=system_prompt,
                        session_key=cli_sk,
                    )
                cli.allow_input()
            else:
                if force_agent_id:
                    aid = force_agent_id
                    a = agent_mgr.get_agent(aid)
                    if a:
                        sk = build_session_key(aid, channel=msg.channel, account_id=msg.account_id,
                                               peer_id=msg.peer_id, dm_scope=a.dm_scope)
                        result = run_async(run_agent(
                            agent_mgr, aid, sk, msg.text,
                            channel=msg.channel,
                            llm_client=llm_client,
                        ))
                        ch = mgr.get(msg.channel)
                        if ch:
                            ch.send(msg.peer_id, result)
                    else:
                        print_warn(f"Agent '{aid}' not found")
                else:
                    aid, sk = resolve_route(bindings, agent_mgr, channel=msg.channel,
                                            account_id=msg.account_id, peer_id=msg.peer_id)
                    result = run_async(run_agent(
                        agent_mgr, aid, sk, msg.text,
                        channel=msg.channel,
                        llm_client=llm_client,
                    ))
                    ch = mgr.get(msg.channel)
                    if ch:
                        ch.send(msg.peer_id, result)
                    else:
                        print_assistant(result)

    except KeyboardInterrupt:
        print()
    finally:
        stop_event.set()
        cli_thread.join(timeout=2.0)
        if gw_server:
            run_async(gw_server.stop())
        mgr.close_all()
        print(f"{DIM}再见.{RESET}")

# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    agent_loop()

if __name__ == "__main__":
    main()
