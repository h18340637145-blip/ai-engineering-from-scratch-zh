# 本地 Ollama 全中文课程实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 使用本地 Ollama 将课程站点界面与英文课程正文翻译为简体中文，并发布经验证的 Vercel 版本。

**架构：** 新增 Python 翻译器，将 Markdown 拆分为可翻译文本块和不可修改的代码、链接及图表块；每个结果写入同目录检查点，成功后原子覆盖源文件。站点构建仍使用 `node site/build.js`，以保留现有 Vercel 设置。

**技术栈：** Python 3、Ollama HTTP API（本机 `qwen3:14b`）、unittest、Node.js、Vercel。

---

### 任务 1：实现可恢复 Markdown 翻译器

**文件：**
- 创建：`scripts/translate_docs_ollama.py`
- 测试：`scripts/tests/test_translate_docs_ollama.py`

- [ ] **步骤 1：编写失败的测试**

```python
def test_protects_fenced_code_and_markdown_links(self):
    parts = split_markdown("说明 [链接](https://example.com)\\n```python\\nprint('hello')\\n```")
    self.assertEqual(parts[1].text, "```python\\nprint('hello')\\n```")
    self.assertFalse(parts[1].translatable)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m unittest scripts.tests.test_translate_docs_ollama -v`

预期：FAIL，提示 `translate_docs_ollama` 尚不存在。

- [ ] **步骤 3：实现最少保护逻辑和 Ollama 客户端**

```python
def split_markdown(text):
    # fenced code、HTML 注释、Mermaid 和 URL 链接均作为不可翻译块保留
    ...

def translate_batch(parts, model):
    # 仅向 http://127.0.0.1:11434/api/generate 发送普通文本块
    ...
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m unittest scripts.tests.test_translate_docs_ollama -v`

预期：PASS。

### 任务 2：翻译正文并审计中文覆盖率

**文件：**
- 修改：`phases/**/docs/en.md`
- 创建：`.translation-state/docs-zh.json`

- [ ] **步骤 1：预检英文课程清单**

运行：`python3 scripts/translate_docs_ollama.py --dry-run --model qwen3:14b`

预期：输出待翻译文件数，不写入课程文件。

- [ ] **步骤 2：逐阶段调用本地模型**

运行：`python3 scripts/translate_docs_ollama.py --model qwen3:14b --checkpoint .translation-state/docs-zh.json`

预期：已完成的文件写入检查点；再次运行跳过这些文件。

- [ ] **步骤 3：审计结构和中文覆盖率**

运行：`python3 scripts/audit_lessons.py && python3 scripts/translate_docs_ollama.py --verify-only`

预期：课程结构通过，报告没有未完成英文正文。

### 任务 3：中文界面构建、验证与发布

**文件：**
- 修改：`site/index.html`、`site/app.js`、`site/lesson.html`、`site/catalog.html`、`site/glossary.html`、`site/prereqs.html`、`site/about.html`
- 修改：`site/data.js`（由构建生成）

- [ ] **步骤 1：将网站交互文案替换为简体中文**

运行：`node site/build.js`

预期：课程目录重新生成，不改变课程链接。

- [ ] **步骤 2：验证构建与页面**

运行：`node site/build.js && python3 scripts/audit_lessons.py && git diff --check`

预期：全部命令退出码为 0；首页和课程页可显示中文。

- [ ] **步骤 3：提交、推送并跟踪 Vercel**

运行：`git push origin codex/site-zh:main`，然后 `vercel inspect <deployment-url>`。

预期：部署状态为 `Ready`，线上首页与一节课程页均显示中文。
