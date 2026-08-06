# Zygote、SystemServer 与 init rc 启动关系模拟
# 课程文档：phases/22-android-framework-system-basics/03-zygote-and-system-server/docs/en.md
# 参考资料：docs/AndroidFramework/Android Framework 基础.md 的 Zygote 启动流程章节
# 仅解析简化 rc 行和派生启动计划，不读取真实 init 配置。

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZygoteRc:
    command: str
    socket_name: str
    starts_system_server: bool

    @classmethod
    def parse(cls, line: str) -> "ZygoteRc":
        fields = line.split()
        if len(fields) < 3 or fields[0] != "service":
            raise ValueError("必须是 service zygote rc 行")
        command = fields[2]
        socket_name = next((field.split("=", 1)[1] for field in fields if field.startswith("--socket-name=")), "")
        return cls(command, socket_name, "--start-system-server" in fields)


class ZygotePlanner:
    @staticmethod
    def spawn(zygote_name: str) -> list[str]:
        return ["init", zygote_name, "system_server", "application process"]

    @staticmethod
    def rc_for(ro_zygote: str) -> str:
        return f"init.{ro_zygote}.rc"


def main() -> None:
    print("=== Zygote 启动计划 ===")
    rc = ZygoteRc.parse("service zygote /system/bin/app_process64 --zygote --start-system-server --socket-name=zygote")
    print("命令：", rc.command)
    print("进程链：", " -> ".join(ZygotePlanner.spawn("zygote64")))
    print("rc 文件：", ZygotePlanner.rc_for("zygote64_32"))


if __name__ == "__main__":
    main()
