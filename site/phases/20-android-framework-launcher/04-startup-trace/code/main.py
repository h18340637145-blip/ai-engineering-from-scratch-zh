# Launcher3 启动链路追踪 - 解析 logcat 提取启动时序与耗时
# 课程文档：phases/20-android-framework-launcher/04-startup-trace/docs/en.md
# 参考：AOSP Launcher.java / LauncherModel.java / LoaderTask.java

"""
解析带时间戳的 LauncherStudy 日志，提取启动调用链并计算各阶段耗时。

实际使用时可将 adb logcat -d | grep LauncherStudy 的输出粘贴到本脚本的
DEMO_LOG 变量，或通过 stdin 读取。
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional


# logcat threadtime 格式中的时间和消息提取
LOG_RE = re.compile(
    r"(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\.(\d{3})\s+\d+\s+\d+\s+[VDIWEF]\s+\S+\s*:\s*(.*)"
)

# 关键启动阶段关键词（按期望调用顺序）
STARTUP_STAGES = [
    "Launcher.onCreate begin",
    "Launcher.setupViews begin",
    "LauncherModel.startLoader",
    "LoaderTask.run begin",
    "LoaderTask.loadWorkspace begin",
    "bindWorkspace",
    "loadAllApps",
    "Launcher.finishBindingItems",
]


@dataclass
class LogLine:
    timestamp_ms: int   # 相对毫秒时间戳（从第一条算起）
    raw_time: str
    message: str


def parse_timestamp_ms(hms: str, ms_str: str) -> int:
    """将 HH:MM:SS + ms 转成毫秒整数（用于计算耗时差）。"""
    time_part = hms.strip().split(" ")[-1]
    h, m, s = [int(x) for x in time_part.split(":")]
    return (h * 3600 + m * 60 + s) * 1000 + int(ms_str)


def parse_log(text: str) -> List[LogLine]:
    """解析 logcat 文本，返回有效日志行列表。"""
    lines: List[LogLine] = []
    base_ms: Optional[int] = None
    for raw in text.splitlines():
        m = LOG_RE.match(raw.strip())
        if not m:
            continue
        abs_ms = parse_timestamp_ms(m.group(1), m.group(2))
        if base_ms is None:
            base_ms = abs_ms
        lines.append(LogLine(
            timestamp_ms=abs_ms - base_ms,
            raw_time=m.group(1) + "." + m.group(2),
            message=m.group(3).strip(),
        ))
    return lines


def extract_startup_chain(lines: List[LogLine]) -> List[tuple]:
    """
    按 STARTUP_STAGES 关键词顺序提取启动调用链。

    返回 [(stage_name, LogLine), ...] 列表，顺序与日志一致。
    """
    result: List[tuple] = []
    used_stages = set()
    for line in lines:
        for stage in STARTUP_STAGES:
            if stage in line.message and stage not in used_stages:
                result.append((stage, line))
                used_stages.add(stage)
                break
    return result


# ── 演示数据 ─────────────────────────────────────────────────────────

DEMO_LOG = """\
08-06 10:00:00.100  1234  1234 D LauncherStudy: Launcher.onCreate begin
08-06 10:00:00.150  1234  1234 D LauncherStudy: Launcher.setupViews begin
08-06 10:00:00.180  1234  1234 D LauncherStudy: LauncherModel.startLoader
08-06 10:00:00.200  1234  1235 D LauncherStudy: LoaderTask.run begin
08-06 10:00:00.210  1234  1235 D LauncherStudy: LoaderTask.loadWorkspace begin
08-06 10:00:00.350  1234  1234 D LauncherStudy: bindWorkspace
08-06 10:00:00.400  1234  1235 D LauncherStudy: loadAllApps
08-06 10:00:00.500  1234  1234 D LauncherStudy: Launcher.finishBindingItems
"""


def main() -> None:
    lines = parse_log(DEMO_LOG)
    chain = extract_startup_chain(lines)

    print("=== Launcher3 启动链路分析 ===\n")
    prev_ms = 0
    for i, (stage, log) in enumerate(chain, 1):
        delta = log.timestamp_ms - prev_ms
        delta_str = f"(耗时 +{delta}ms)" if i > 1 else ""
        print(f"  阶段 {i}: {stage:<35} @ {log.raw_time} {delta_str}")
        prev_ms = log.timestamp_ms

    if chain:
        total = chain[-1][1].timestamp_ms - chain[0][1].timestamp_ms
        print(f"\n总启动耗时: {total}ms")
    else:
        print("未找到任何启动日志，请检查输入。")


if __name__ == "__main__":
    main()
