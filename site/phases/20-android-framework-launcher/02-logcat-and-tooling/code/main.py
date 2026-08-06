# Logcat 过滤与分析工具
# 课程文档：phases/20-android-framework-launcher/02-logcat-and-tooling/docs/en.md
# 参考：Android 开发者文档 - Logcat 命令行工具

"""
模拟 adb logcat -s TAG:LEVEL 的核心过滤逻辑。

logcat 输出格式（threadtime 模式）：
  MM-DD HH:MM:SS.mmm PID TID LEVEL TAG: MESSAGE

本脚本解析该格式，并支持：
1. 按 Tag 过滤
2. 按最低日志等级过滤
3. 统计各等级数量
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Dict


LEVEL_PRIORITY: Dict[str, int] = {
    "V": 0, "D": 1, "I": 2, "W": 3, "E": 4, "F": 5
}

# 标准 logcat threadtime 格式正则
LOGCAT_PATTERN = re.compile(
    r"^(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+(\d+)\s+(\d+)\s+([VDIWEF])\s+([\w./-]+)\s*:\s*(.*)$"
)


@dataclass
class LogEntry:
    timestamp: str
    pid: int
    tid: int
    level: str
    tag: str
    message: str

    @classmethod
    def parse(cls, line: str) -> Optional["LogEntry"]:
        """解析一行 logcat 输出，失败时返回 None。"""
        m = LOGCAT_PATTERN.match(line.strip())
        if not m:
            return None
        return cls(
            timestamp=m.group(1),
            pid=int(m.group(2)),
            tid=int(m.group(3)),
            level=m.group(4),
            tag=m.group(5).strip(),
            message=m.group(6),
        )

    def priority(self) -> int:
        return LEVEL_PRIORITY.get(self.level, -1)


class LogcatFilter:
    """模拟 adb logcat -s TAG:LEVEL 过滤行为。"""

    def __init__(self, tag: str, min_level: str = "D") -> None:
        self.tag = tag
        self.min_level = min_level
        self.min_priority = LEVEL_PRIORITY.get(min_level, 1)

    def match(self, entry: LogEntry) -> bool:
        return (
            entry.tag == self.tag
            and entry.priority() >= self.min_priority
        )


def parse_logcat(text: str) -> List[LogEntry]:
    """解析多行 logcat 文本，返回成功解析的 LogEntry 列表。"""
    entries: List[LogEntry] = []
    for line in text.splitlines():
        entry = LogEntry.parse(line)
        if entry:
            entries.append(entry)
    return entries


def filter_by_tag(entries: List[LogEntry], tag: str, min_level: str = "D") -> List[LogEntry]:
    f = LogcatFilter(tag, min_level)
    return [e for e in entries if f.match(e)]


def count_by_level(entries: List[LogEntry]) -> Dict[str, int]:
    counts: Dict[str, int] = {lv: 0 for lv in LEVEL_PRIORITY}
    for e in entries:
        if e.level in counts:
            counts[e.level] += 1
    return counts


# ── 演示数据 ─────────────────────────────────────────────────────────

DEMO_LOGCAT = """\
08-06 10:00:00.100  1234  1234 D LauncherStudy: Launcher.onCreate begin
08-06 10:00:00.150  1234  1234 D LauncherStudy: Launcher.setupViews begin
08-06 10:00:00.200  1234  1235 D LoaderTask    : LoaderTask.run begin
08-06 10:00:00.210  1234  1235 D LauncherStudy: LauncherModel.startLoader
08-06 10:00:00.300  1234  1235 D LauncherStudy: LoaderTask.loadWorkspace begin
08-06 10:00:00.320  1234  1235 D LoaderCursor  : read row id=1 type=0
08-06 10:00:00.321  1234  1235 W LoaderCursor  : item overlap detected at (0,0)
08-06 10:00:00.500  1234  1234 D LauncherStudy: Launcher.finishBindingItems
08-06 10:00:00.510  1234  1234 I System.out    : Launcher ready
08-06 10:00:01.000  1234  1234 D InputEventReceiver: ACTION_MOVE
08-06 10:00:01.001  1234  1234 D InputEventReceiver: ACTION_MOVE
08-06 10:00:01.002  1234  1234 E LauncherStudy: unexpected null itemInfo
"""


def main() -> None:
    entries = parse_logcat(DEMO_LOGCAT)

    print("=== 过滤 Tag: LauncherStudy（最低等级 D）===\n")
    filtered = filter_by_tag(entries, "LauncherStudy", "D")
    for e in filtered:
        print(f"  {e.level}  {e.tag}: {e.message}")

    print("\n=== 全量日志等级统计 ===\n")
    counts = count_by_level(entries)
    for lv in ["V", "D", "I", "W", "E"]:
        print(f"  {lv}: {counts[lv]} 条")

    print("\n=== 排除 ACTION_MOVE 噪声后的全量日志 ===\n")
    clean = [e for e in entries if "ACTION_MOVE" not in e.message]
    for e in clean:
        print(f"  {e.level}  {e.tag}: {e.message}")


if __name__ == "__main__":
    main()
