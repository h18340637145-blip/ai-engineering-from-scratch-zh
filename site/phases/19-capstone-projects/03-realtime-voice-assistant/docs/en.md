# 第 03 章 — 实时语音助手（ASR 到 LLM 再到 TTS）

> 一个真正好用的语音智能体，端到端延迟必须低于 800ms，要能判断用户什么时候说完了、能处理中途插话，还要能调用工具而不把整条链路卡住。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都达到了这个门槛。它们的形态也很一致：流式 ASR、轮次检测器、流式 LLM 和流式 TTS，通过 WebRTC 串起来，并且每一跳都把延迟预算压得很紧。把它完整做出来，测 WER、MOS 和误截断率，再在丢包环境下跑起来。

**Type:** Capstone
**Languages:** Python（智能体 + 流水线）、TypeScript（Web 客户端）
**Prerequisites:** 第 6 章（语音与音频）、第 7 章（Transformer）、第 11 章（LLM 工程）、第 13 章（工具）、第 14 章（智能体）、第 17 章（基础设施）
**Phases exercised:** P6 · P7 · P11 · P13 · P14 · P17
**Time:** 30 小时

## 问题

2025 到 2026 年，语音一直是 AI 交互里变化最快的赛道。技术上限一季比一季高。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70，都已经把首字音频输出压到了 800ms 以内。门槛并不只有延迟，更是交互体感：不能打断用户、不能被用户打断、要能在一句话中途被插入打断后恢复、要能在对话中间调用工具而不让音频停住、还要能扛住抖动严重的移动网络。

靠拼三次 REST 调用到不了这个级别。架构必须从头到尾都做成分段流式。真正做起来以后，失败模式也会变得非常直观：为电话音频调好的 VAD 却被背景电视误触发、轮次检测器一直等永远不会出现的标点、TTS 在发声前先缓冲了 400ms。这个阶段项目的目标，就是在负载下把这些问题一个个修掉，并输出一份延迟与质量报告。

## 概念

这条流水线有五个流式阶段：**音频输入**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转写）、**轮次检测**（VAD 加上一个读取部分转写、判断结束信号的小型轮次检测模型）、**LLM**（一旦判断用户说完，就立刻开始流式输出 token）、**TTS**（在第一个 LLM token 之后约 200ms 内开始流式输出音频）。

还有三个横切关注点。**插话抢占**：当用户在智能体说话时开始发言，TTS 立即取消，ASR 立刻接管。**工具使用**：对话中的函数调用（天气、日历）必须走旁路，不能拖慢音频；如果工具超过 300ms 还没返回，智能体要先补一个确认 token（“稍等一下……”）。**背压**：在丢包条件下，部分转写会被暂存，VAD 会提高语音门限，智能体也会避免在未确认消息上方继续说话。

衡量标准是量化的。Hamming VAD 基准在 15 dB SNR 下的 WER 要低于 8%。100 次实测调用里的首字音频输出 p50 要低于 800ms。误截断率要低于 3%。TTS 的 MOS 要高于 4.2。单台 g5.xlarge 要能支撑 50 路并发呼叫。这些数字就是交付物本身。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## Stack

- 传输层：LiveKit Agents 1.0（WebRTC）外加 Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- ASR：Deepgram Nova-3（流式，首个部分结果低于 300ms）或自托管的 faster-whisper Whisper-v3-turbo
- VAD：Silero VAD v5 加上 LiveKit 轮次检测器（读取部分转写的小型 Transformer）
- LLM：OpenAI GPT-4o-realtime 用于紧密集成，Gemini 2.5 Flash Live，或级联式 Claude Haiku 4.5（流式补全，独立音频路径）
- TTS：Cartesia Sonic-2（首字节最快）、ElevenLabs Flash v3，或自托管开源 Orpheus
- 工具：通过 FastMCP 旁路提供天气 / 日历 / 预订；如果工具耗时超过 300ms，智能体先吐出填充语
- 可观测性：OpenTelemetry 语音 span，支持音频回放的 Langfuse 语音 trace
- 部署：单台 g5.xlarge（24GB VRAM）用于自托管 Whisper + Orpheus；若要最低延迟则用托管 API

## Build It

1. **WebRTC 会话。** 搭起一个 LiveKit 房间和一个负责上传麦克风音频的 Web 客户端。服务端挂上一个会加入房间的智能体 worker。

2. **ASR 流式处理。** 将 20ms 的 PCM 帧送给 Deepgram Nova-3（或者 GPU 上的 faster-whisper）。订阅部分转写和最终转写，并记录每个部分结果的延迟。

3. **VAD 与轮次检测。** 在帧流上运行 Silero VAD v5。检测到语音结束事件后，用最新的部分转写去触发 LiveKit 轮次检测器。只有当 VAD 判定静音持续 500ms 且轮次检测器的结束分数大于 0.6 时，才真正提交“轮次结束”。

4. **LLM 流式输出。** 一旦轮次结束，就把当前对话和最终转写一起送进 LLM 调用，并持续流式输出 token。拿到第一个 token 后，立刻交给 TTS。

5. **TTS 流式输出。** Cartesia Sonic-2 返回音频分块。第一个分块必须在第一个 LLM token 之后 200ms 内离开服务端。把分块发进 LiveKit 房间；客户端通过 WebRTC jitter buffer 播放。

6. **插话抢占。** 当 VAD 在 TTS 播放期间检测到新的用户语音时，立即取消 TTS 流，丢弃剩余的 LLM 输出，并重新武装 ASR。发布一个 `tts_canceled` span。

7. **工具旁路。** 将天气和日历注册为函数调用工具。触发时并发发起调用；如果 300ms 内还没返回，让 LLM 先输出“稍等一下，我看一眼”；工具返回后继续。

8. **评测框架。** 录制 100 通电话。计算 WER（对照留出集转写）、误截断率（用户还在说话时 TTS 被取消）、首字音频输出 p50、TTS MOS（人工或 NISQA）以及一次抖动丢包测试（随机丢弃 3% 的包）。

9. **负载测试。** 用一个合成呼叫器在单台 g5.xlarge 上压 50 路并发。测持续性的首字音频输出 p95。

## 使用示例

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付

`outputs/skill-voice-agent.md` 就是最终交付物。给定一个场景（客服、日程安排或自助终端），它会搭起一个 LiveKit 智能体，并把 ASR / VAD / LLM / TTS 流水线调到上面的量化门槛。评分标准如下：

| 权重 | 标准 | 评测方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 100 次录音调用里的首字音频输出 p50 低于 800ms |
| 20 | 轮次质量 | Hamming VAD 基准上的误截断率低于 3% |
| 20 | 工具调用正确性 | 对话中的工具调用能返回正确数据，同时不会拖慢音频 |
| 20 | 丢包下的可靠性 | 注入 3% 丢包后，WER 和轮次稳定性仍然可接受 |
| 15 | 评测框架完整性 | 使用公开配置即可复现全部测量 |
| **100** | | |

## 练习

1. 在 g5.xlarge 上把 Deepgram Nova-3 换成 faster-whisper v3 turbo。测延迟和 WER 差距，找出 CPU 与 GPU 决策真正起作用的地方。

2. 加入一个插话仲裁策略：当用户在工具调用期间插话时，智能体应该怎么做？比较三种策略（强制取消、工具完成后再停、排队到下一轮）。

3. 做一次对抗性的轮次检测测试：让用户在句中停顿很久。调 VAD 静音阈值和轮次检测器分数阈值，在不超过 900ms 的前提下把误截断压到最低。

4. 通过 Twilio 把同一个智能体部署到 PSTN。比较 PSTN 和 WebRTC 的首字音频输出，并解释 jitter buffer 和编解码器差异。

5. 为非英语语言（如日语、西班牙语）加入语音活动检测。测 Silero VAD v5 的误触发率，并和针对语言的微调模型做比较。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 轮次检测 | “话轮结束” | 给定 VAD 静音和部分转写，判断用户是不是已经说完的分类器 |
| 插话抢占 | “打断处理” | 当 VAD 检测到新的用户语音时，中途取消 TTS 播放 |
| 首字音频输出 | “延迟” | 从用户停止说话到第一个音频包离开服务端之间的时间 |
| VAD | “语音门” | 把音频帧分成语音或静音的模型；Silero VAD v5 是 2026 年默认选择 |
| Jitter buffer | “音频平滑” | 客户端侧的缓冲区，用来短暂缓存数据包以吸收网络抖动 |
| 填充语 | “确认 token” | 工具太慢时，智能体发出的短句，用来避免长时间沉默 |
| MOS | “主观评分” | 语音质量的感知评分；NISQA 是自动化代理指标 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — WebRTC 智能体参考框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 另一套 Python 优先的流式智能体框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型的参考
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI architecture](https://docs.retellai.com) — 生产级语音智能体架构
- [Vapi.ai production stack](https://docs.vapi.ai) — 另一套生产参考
