# Handler、Looper 与 MessageQueue 模拟
# 课程文档：phases/21-java-android-foundations/05-handler-looper-and-concurrency/docs/en.md
# 参考资料：docs/AndroidFramework/Java android .md 的 Handler、Looper、MessageQueue 章节
# 以确定性的时间戳队列表达延迟消息与页面销毁时移除回调。

from __future__ import annotations

from dataclasses import dataclass


@dataclass(order=True)
class Message:
    when_ms: int
    name: str


class MessageQueueSimulator:
    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._loopers = {"main"}

    def post(self, name: str, now_ms: int, delay_ms: int = 0) -> None:
        self._messages.append(Message(now_ms + delay_ms, name))
        self._messages.sort()

    def next_due(self, now_ms: int) -> Message | None:
        if not self._messages or self._messages[0].when_ms > now_ms:
            return None
        return self._messages.pop(0)

    def remove(self, name: str) -> None:
        self._messages = [message for message in self._messages if message.name != name]

    def has_looper(self, thread_name: str) -> bool:
        return thread_name in self._loopers

    def prepare(self, thread_name: str) -> None:
        self._loopers.add(thread_name)


def main() -> None:
    print("=== Handler 消息队列 ===")
    queue = MessageQueueSimulator()
    queue.post("延迟渲染", now_ms=0, delay_ms=16)
    queue.post("立即刷新", now_ms=0)
    print("0 ms：", queue.next_due(0))
    print("16 ms：", queue.next_due(16))
    print("worker 有 Looper：", queue.has_looper("worker"))


if __name__ == "__main__":
    main()
