# Android 启动链与 bootanimation 规格解析
# 课程文档：phases/22-android-framework-system-basics/01-boot-chain-and-bootanimation/docs/en.md
# 参考资料：docs/AndroidFramework/Android Framework 基础.md 的系统启动与开机动画章节
# 仅解析 desc.txt 和事件时间线，不读取设备文件或操作系统分区。

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnimationPart:
    mode: str
    count: int
    pause: int
    path: str

    @property
    def interruptible(self) -> bool:
        return self.mode == "p"


@dataclass(frozen=True)
class BootAnimationSpec:
    width: int
    height: int
    fps: int
    parts: list[AnimationPart]

    @classmethod
    def parse(cls, text: str) -> "BootAnimationSpec":
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if not lines:
            raise ValueError("desc.txt 不能为空")
        header = lines[0].split()
        if len(header) != 3:
            raise ValueError("首行必须是宽 高 帧率")
        try:
            width, height, fps = (int(value) for value in header)
        except ValueError as exc:
            raise ValueError("宽、高和帧率必须为整数") from exc
        parts: list[AnimationPart] = []
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 4 or fields[0] not in {"p", "c"}:
                raise ValueError(f"无效动画段：{line}")
            parts.append(AnimationPart(fields[0], int(fields[1]), int(fields[2]), fields[3]))
        return cls(width, height, fps, parts)


class BootTimeline:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def find_gap(self) -> int | None:
        stop_time = done_time = None
        for event in self.events:
            pieces = event.split(maxsplit=1)
            if len(pieces) != 2 or not pieces[0].isdigit():
                continue
            timestamp, message = int(pieces[0]), pieces[1]
            if "stop_bootanim" in message:
                stop_time = timestamp
            if "wm_boot_animation_done" in message:
                done_time = timestamp
        if stop_time is None or done_time is None:
            return None
        return done_time - stop_time


def main() -> None:
    print("=== bootanimation 规格与时间线 ===")
    spec = BootAnimationSpec.parse("1080 1920 30\np 1 0 part0\np 0 0 part1")
    print(f"规格：{spec.width}×{spec.height} @ {spec.fps} fps")
    print("首段可中断：", spec.parts[0].interruptible)
    print("结束间隔：", BootTimeline(["10 stop_bootanim", "17 wm_boot_animation_done"]).find_gap())


if __name__ == "__main__":
    main()
