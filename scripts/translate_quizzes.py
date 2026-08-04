#!/usr/bin/env python3
"""Translate quiz.json files from English to Chinese using Google Translate (free)."""
import json
import pathlib
import time

from deep_translator import GoogleTranslator

ROOT = pathlib.Path(__file__).parent.parent
PHASES = ROOT / "phases"

_translator = GoogleTranslator(source="en", target="zh-CN")

# Max chars per batch for Google Translate
BATCH_LIMIT = 4500


def translate_strings(texts: list[str]) -> list[str]:
    """Translate a list of strings, batching to stay within API limits."""
    if not texts:
        return []
    results: list[str] = []
    batch: list[str] = []
    batch_len = 0
    batch_start = 0

    def flush():
        nonlocal batch, batch_len, batch_start
        if not batch:
            return
        translated = _translator.translate_batch(batch)
        results.extend(t if t else batch[i] for i, t in enumerate(translated))
        batch = []
        batch_len = 0

    for text in texts:
        if batch_len + len(text) + 1 > BATCH_LIMIT and batch:
            flush()
        batch.append(text)
        batch_len += len(text) + 1
    flush()
    return results


def has_english(obj) -> bool:
    """Return True if the question fields look like English (not Chinese)."""
    questions = []
    if isinstance(obj, dict) and "questions" in obj:
        questions = obj["questions"]
    elif isinstance(obj, list):
        questions = obj
    if not questions:
        return False
    for q in questions[:2]:  # check first two questions
        if not isinstance(q, dict):
            continue
        text = q.get("question", "")
        # Chinese characters indicate already translated
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
        if ascii_letters > 5 and chinese_chars < 3:
            return True
    return False


def translate_file(path: pathlib.Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  SKIP (invalid JSON): {path.relative_to(ROOT)}")
        return False

    if not has_english(data):
        return False  # already translated

    # Extract text fields preserving positions
    questions = data["questions"] if isinstance(data, dict) else data
    strings: list[str] = []
    positions: list[tuple] = []  # (q_idx, field, opt_idx)

    # Also translate top-level title
    top_title = None
    if isinstance(data, dict) and "title" in data:
        top_title = data["title"]
        ascii_l = sum(1 for c in top_title if c.isascii() and c.isalpha())
        if ascii_l > 3:
            strings.append(top_title)
            positions.append(("title", None, None))

    for qi, q in enumerate(questions):
        for field in ("question", "explanation"):
            val = q.get(field, "")
            if val and sum(1 for c in val if c.isascii() and c.isalpha()) > 3:
                strings.append(val)
                positions.append((qi, field, None))
        for oi, opt in enumerate(q.get("options", [])):
            if sum(1 for c in opt if c.isascii() and c.isalpha()) > 3:
                strings.append(opt)
                positions.append((qi, "options", oi))

    if not strings:
        return False

    try:
        translated = translate_strings(strings)
    except Exception as e:
        print(f"  TRANSLATE ERROR: {e}")
        return False

    # Reassemble
    for pos, tr in zip(positions, translated):
        if pos[0] == "title":
            data["title"] = tr
        else:
            qi, field, oi = pos
            q = questions[qi]
            if field == "options":
                q["options"][oi] = tr
            else:
                q[field] = tr

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def main():
    files = sorted(PHASES.rglob("quiz.json"))
    to_translate = [f for f in files if has_english(json.loads(f.read_text())
                    if f.read_text().strip() else {})]

    total = len(to_translate)
    print(f"Found {total} quiz.json files with English content to translate")

    done = 0
    errors = 0
    for i, path in enumerate(to_translate, 1):
        rel = path.relative_to(ROOT)
        print(f"[{i}/{total}] {rel} ... ", end="", flush=True)
        try:
            ok = translate_file(path)
            if ok:
                print("OK")
                done += 1
            else:
                print("skipped")
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1
        time.sleep(0.3)

    print(f"\nDone: {done} translated, {errors} errors out of {total} files")


if __name__ == "__main__":
    main()
