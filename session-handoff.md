# 会话交接：AI Engineering from Scratch（中文工作区）

**更新日期：** 2026-08-06  
**当前分支：** `main`  
**工作区状态：** 本次会话新增 `docs/product.md`、`docs/architecture.md` 与本文件；开始后请先运行 `git status --short` 确认是否还有其他协作者的未提交修改。

## 本次完成事项

- 分析了课程内容、静态站点、自动化校验、图书构建和技能安装流程。
- 新增产品说明：`docs/product.md`。
- 新增架构说明：`docs/architecture.md`。
- 新增本交接文档，供下一次会话快速定位项目边界和操作规范。

## 当前项目事实

以 `python3 scripts/build_catalog.py --stdout` 的实时扫描结果为准：

| 指标 | 数量 |
|---|---:|
| 阶段 | 21 |
| 课程 | 513 |
| 代码文件 | 764 |
| Skills | 398 |
| Prompts | 99 |
| Agents | 0 |

README 也宣称 513 节课程、21 个阶段，表明公开入口与目录扫描当前一致。

## 关键入口

| 目的 | 位置 |
|---|---|
| 贡献规则与课程契约 | `AGENTS.md` |
| 课程公开入口与阶段表 | `README.md` |
| 课程完成状态 | `ROADMAP.md` |
| 课程主事实源 | `phases/<phase>/<lesson>/` |
| 课程结构审计 | `scripts/audit_lessons.py` |
| 从目录生成统计/清单 | `scripts/build_catalog.py` |
| README 计数检查 | `scripts/check_readme_counts.py` |
| 安装课程产物 | `scripts/install_skills.py` |
| 静态网站数据构建 | `site/build.js` |
| 静态部署配置 | `vercel.json` |
| 图书构建 | `scripts/build_book.py` |
| 课程 CI | `.github/workflows/curriculum.yml` |
| 图书 CI | `.github/workflows/build-book.yml` |

## 下一位协作者的工作方式

### 修改或新增课程

1. 仅在目标课程目录中修改 `docs/en.md`、`code/`、`quiz.json`、`outputs/` 等必要文件。
2. 新课程需要补充 README 的 Markdown 链接行，并按需要更新 ROADMAP 与术语表。
3. 保持 `AGENTS.md` 的课程契约：文档 H1、可运行代码、标准 quiz schema、至少 5 个测试，以及受限依赖。
4. 先运行目标课程的主程序和测试，再运行仓库级审计。
5. 提交时保持“一课一提交”，使用长度不超过 72 字符的 Conventional Commit 标题。

### 修改站点

1. 内容列表、状态或术语变更应先改 `README.md`、`ROADMAP.md`、`glossary/terms.md`，再运行 `node site/build.js` 检查解析结果。
2. 不要手工维护 `catalog.json`，它是 gitignored 的临时生成物。
3. 常规课程 PR 不手改 `site/data.js`；主分支 CI 会在 README 计数同步后自动重建并提交它。
4. 对页面行为改动，检查 `site/index.html`/`app.js`、`lesson.html` 及对应的样式或数据依赖。

### 构建图书或安装产物

```bash
# 只拼装或生成 EPUB（需要 Pandoc）
python3 scripts/build_book.py

# 同时生成 PDF（需要 Pandoc 和 XeLaTeX）
python3 scripts/build_book.py --pdf

# 预览可安装技能，不写文件
python3 scripts/install_skills.py /tmp/aiefs-skills --dry-run
```

## 验证清单

```bash
# 所有课程目录不变量
python3 scripts/audit_lessons.py

# README 自动计数是否漂移
python3 scripts/check_readme_counts.py

# 重新生成站点数据；仅用于本地验证或站点变更
node site/build.js

# 查看本地修改，注意不要撤销其他协作者的工作
git status --short
git diff --check
```

## 容易踩到的约束

- README 课程表若缺少 Markdown 链接，`site/build.js` 无法生成该课程的站点路径。
- ROADMAP 的 `✅`、`🚧`、`⬚` 和表格结构是网站解析输入。
- `site/data.js`、README 计数由主分支 CI 派生；`catalog.json` 不提交。
- 课程文档的所有 fenced code block 要标语言；图表只使用 Mermaid 或 SVG。
- 不引入允许列表之外的依赖。优先标准库，以维持教学可读性。
- 不执行 `git reset --hard` 或其他可能丢失协作者改动的命令。

## 待定事项

本次任务仅要求项目分析与文档落地，未发现需要立即修复的失败项，也没有启动开发服务器或修改课程内容。下一项具体工作应以新需求为准。