import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class MarkdownProtectionTests(unittest.TestCase):
    def test_translate_file_does_not_send_links_or_inline_code_to_ollama(self):
        from scripts.translate_docs_ollama import translate_file

        source = (
            "This lesson explains how to build an AI agent. It introduces planning, memory, tools, and evaluation. "
            "The paragraph is intentionally long enough to trigger translation. "
            "It also uses [课程链接](https://example.com/course) and `inline code`.\n\n"
            "```python\nprint('hello')\n```\n"
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "en.md"
            path.write_text(source, encoding="utf-8")

            seen_inputs: list[str] = []

            def fake_translate(text: str, model: str) -> str:
                seen_inputs.append(text)
                return f"译文::{text}"

            with patch("scripts.translate_docs_ollama.ollama_translate", side_effect=fake_translate):
                translate_file(path, "qwen3:14b")

            translated = path.read_text(encoding="utf-8")

        self.assertTrue(seen_inputs, "应该至少翻译一个正文块")
        self.assertNotIn("https://example.com/course", "".join(seen_inputs))
        self.assertNotIn("`inline code`", "".join(seen_inputs))
        self.assertIn("[课程链接](https://example.com/course)", translated)
        self.assertIn("`inline code`", translated)
        self.assertIn("```python\nprint('hello')\n```", translated)


if __name__ == "__main__":
    unittest.main()
