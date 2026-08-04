# Git 与 AI 项目协作

> 版本控制不仅针对代码——也关乎模型、实验与大型数据集。掌握 Git 工作流、Git LFS 以及团队提交规范。

**Type:** 构建
**Languages:** Bash, Python
**Prerequisites:** 无
**Time:** ~30 分钟

## 学习目标

- 配置 Git 身份并使用日常工作流中的添加（add）、提交（commit）和推送（push）
- 为独立实验创建和合并分支，而不会破坏主分支（main）
- 编写一个 `.gitignore` 以排除模型检查点和大型二进制文件
- 使用 `git log` 浏览提交历史，以理解项目的发展过程

## 问题

你即将编写跨越 20 个阶段的数百个代码文件。如果没有版本控制，你将会丢失工作、破坏无法恢复的内容，并且没有与他人协作的方法。

Git 是工具。GitHub 是代码存放的地方。本课程涵盖你完成本课程所需的内容，仅此而已。

## 概念 /no_think

<>

## 概念```mermaid
sequenceDiagram
    participant WD as Working Directory
    participant SA as Staging Area
    participant LR as Local Repo
    participant R as Remote (GitHub)
    WD->>SA: git add
    SA->>LR: git commit
    LR->>R: git push
    R->>LR: git fetch
    LR->>WD: git pull
```需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think

<>

需要记住的三件事：
1. 经常保存（`git commit`）
2. 推送至远程仓库（`git push`）
3. 使用分支进行实验（`git checkout -b experiment`）

## 构建它

### 步骤1：配置git

  /no_think```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```### 步骤 2：日常流程```bash
git status
git add file.py
git commit -m "Add perceptron implementation"
git push origin main
```### 步骤 3：为实验创建分支```bash
git checkout -b experiment/new-optimizer

# ... make changes, commit ...

git checkout main
git merge experiment/new-optimizer
```### 步骤4：使用该课程仓库

您无法直接向课程仓库推送代码——只有维护者拥有写入权限。首先在GitHub上对其进行 Fork（Fork 按钮，右上角），以便 `origin` 指向您自己的副本：```bash
git clone https://github.com/YOUR-USERNAME/ai-engineering-from-scratch.git
cd ai-engineering-from-scratch

git checkout -b my-progress
# work through lessons, commit your code
git push origin my-progress
```## 使用方法

对于本课程，你需要使用以下命令：

| 命令 | 使用场景 |
|---------|------|
| `git clone` | 获取课程仓库 |
| `git add` + `git commit` | 保存你的工作 |
| `git push` | 备份到 GitHub |
| `git checkout -b` | 在不破坏主分支的情况下尝试新内容 |
| `git log --oneline` | 查看你的操作记录 |

仅此而已。本课程不需要使用 rebase、cherry-pick 或 submodules。

## 练习

1. Fork 这个仓库，克隆你的 Fork，创建一个名为 `my-progress` 的分支，新建一个文件，提交更改，推送至远程仓库
2. 创建一个 `.gitignore` 文件，排除模型检查点文件（`.pt`、`.pth`、`.safetensors`）
3. 使用 `git log --oneline` 查看本仓库的提交历史，阅读课程是如何逐步添加的

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 提交 | "保存" | 某个时间点整个项目的快照 |
| 分支 | "一个副本" | 指向某个提交的指针，随着工作推进会向前移动 |
| 合并 | "合并代码" | 从一个分支获取更改并应用到另一个分支 |
| 远程仓库 | "云端" | 存储在其他位置（GitHub、GitLab）的仓库副本 |
