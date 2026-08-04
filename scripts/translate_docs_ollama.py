#!/usr/bin/env python3
"""使用本机 Ollama 将课程 Markdown 正文翻译为简体中文。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / ".translation-state" / "docs-zh.json"

BLOCK_PATTERN = re.compile(r"(```[\s\S]*?```|<!--[\s\S]*?-->)")
INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")
MARKDOWN_LINK_TARGET_PATTERN = re.compile(r"(!?\[[^\]]*\]\()([^)]+)(\))")
BARE_URL_PATTERN = re.compile(r"(?<!\()(?P<url>https?://[^\s<>\")]+)")
PLACEHOLDER_PATTERN = re.compile(r"@@(TOKEN|URL)(\d+)@@")
THINK_BLOCK_PATTERN = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


@dataclass
class Part:
    text: str
    translatable: bool


def split_markdown(text: str) -> list[Part]:
    """保留围栏代码、HTML 注释等不可翻译块。"""
    parts: list[Part] = []
    for segment in BLOCK_PATTERN.split(text):
        if not segment:
            continue
        protected = segment.startswith("```") or segment.startswith("<!--")
        parts.append(Part(segment, not protected))
    return parts


def mask_inline_tokens(text: str) -> tuple[str, dict[str, str]]:
    """把 URL、Markdown 链接目标和内联代码替换为占位符。"""
    tokens: dict[str, str] = {}
    counter = {"value": 0}

    def next_placeholder(prefix: str, value: str) -> str:
        placeholder = f"@@{prefix}{counter['value']}@@"
        counter["value"] += 1
        tokens[placeholder] = value
        return placeholder

    def replace_markdown_target(match: re.Match[str]) -> str:
        return f"{match.group(1)}{next_placeholder('URL', match.group(2))}{match.group(3)}"

    masked = INLINE_CODE_PATTERN.sub(lambda match: next_placeholder("TOKEN", match.group(0)), text)
    masked = MARKDOWN_LINK_TARGET_PATTERN.sub(replace_markdown_target, masked)
    masked = BARE_URL_PATTERN.sub(lambda match: next_placeholder("URL", match.group("url")), masked)
    return masked, tokens


def restore_inline_tokens(text: str, tokens: dict[str, str]) -> str:
    """把占位符恢复成原始 token。"""
    return PLACEHOLDER_PATTERN.sub(lambda match: tokens[match.group(0)], text)


def ollama_translate(text: str, model: str) -> str:
    prompt = (
        "将以下 Markdown 文本完整翻译为简体中文。"
        "保留 Markdown 标记、段落、列表、标题层级、占位符、变量名和专有技术名词；"
        "不要省略任何标题，不要输出解释，不要输出思考过程，只输出译文。\n\n"
        + text
    )
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False, "think": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.loads(response.read())
    translated = payload["response"].strip()
    translated = THINK_BLOCK_PATTERN.sub("", translated)
    return translated.replace("/think", "").strip()


def has_english(text: str) -> bool:
    latin = sum(char.isascii() and char.isalpha() for char in text)
    han = sum("\u4e00" <= char <= "\u9fff" for char in text)
    return latin > 80 and latin > han * 3


def translate_fragment(text: str, model: str) -> str:
    masked, tokens = mask_inline_tokens(text)
    translated = ollama_translate(masked, model)
    return restore_inline_tokens(translated, tokens)


def translated_text(source: str, model: str) -> str:
    translated_parts: list[str] = []
    for part in split_markdown(source):
        if part.translatable and part.text.strip():
            translated_parts.append(translate_fragment(part.text, model))
        else:
            translated_parts.append(part.text)
    output = "".join(translated_parts)
    return output if output.endswith("\n") else output + "\n"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("completed", {}))


def save_checkpoint(path: Path, completed: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps({"completed": completed}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def translate_file(path: Path, model: str) -> bool:
    source = path.read_text(encoding="utf-8")
    if not has_english(source):
        return False
    output = translated_text(source, model)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(output, encoding="utf-8")
    os.replace(tmp_path, path)
    return True


def iter_target_paths(phase: str | None) -> list[Path]:
    paths = sorted(ROOT.glob("phases/**/docs/en.md"))
    if phase:
        paths = [path for path in paths if path.parts[-4].startswith(phase)]
    return paths


def iter_pending_paths(paths: Iterable[Path], completed: set[str]) -> list[Path]:
    pending: list[Path] = []
    for path in paths:
        key = str(path.relative_to(ROOT))
        if key in completed:
            continue
        if has_english(path.read_text(encoding="utf-8")):
            pending.append(path)
    return pending


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--phase")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    completed = load_checkpoint(checkpoint_path)
    paths = iter_target_paths(args.phase)
    pending = iter_pending_paths(paths, set(completed))

    print(f"待翻译课程：{len(pending)}")
    if args.dry_run:
        return 0
    if args.verify_only:
        return 0 if not pending else 1

    for index, path in enumerate(pending, 1):
        rel_path = str(path.relative_to(ROOT))
        source_hash = file_hash(path)
        print(f"[{index}/{len(pending)}] {rel_path}", flush=True)
        translate_file(path, args.model)
        completed[rel_path] = source_hash
        save_checkpoint(checkpoint_path, completed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
