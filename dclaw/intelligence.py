from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_ID = os.getenv("MODEL_ID", "MiniMax-M2.7")
WORKDIR = Path(__file__).resolve().parents[1]

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
        f"- Channel: {channel}\n- Current time: {now}\n- Prompt mode: {mode}\n"
        f"- Project root: {WORKDIR}\n"
        f"- Tool workspace: {WORKDIR / 'workspace-main'}"
    )
    sections.append(
        "## Tool Truthfulness\n\n"
        "- File tools are scoped to the Tool workspace above.\n"
        "- When asked to show the working directory, use list_directory on `.` and report only the returned result.\n"
        "- Do not claim a file or directory exists unless the latest tool result confirms it.\n"
        "- If a tool returns an error, denial, or empty result, say so instead of filling gaps from memory or prior conversation."
    )
    sections.append(
        "## Tool Selection Rules\n\n"
        "- List a directory: use list_directory.\n"
        "- Find files by name, extension, or path pattern: use glob.\n"
        "- Search inside files for code, errors, symbols, or text: use grep.\n"
        "- Read a known file path: use read_file.\n"
        "- Modify files: use edit_file for targeted replacements and write_file for complete file writes.\n"
        "- Run tests, git, package managers, or project commands: use bash.\n"
        "- Search current or external web information: use web_search and include source URLs in the answer.\n"
        "- Fetch a specific web page from a search result or user-provided URL: use web_fetch.\n"
        "- Do not use bash for ordinary directory listing, file discovery, file reading, or text search."
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

