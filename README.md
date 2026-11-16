# Voice Clone Realtime Translator（独立项目）

一个完全独立的实时同声传译 + 音色克隆项目，参考 **Split Pipeline（VAD → 流式 ASR → 术语增强 MT → 音色克隆流式 TTS → 虚拟声卡输出）** 架构设计。

> 与现有「获客系统」**完全隔离**：独立目录、独立虚拟环境、独立配置与端口（默认 `8900`），无共享数据库或状态冲突。

## 目录结构

```text
voice-clone-realtime-translator/
├── README.md
├── pyproject.toml
├── .env.example
└── src/
    ├── __init__.py
    ├── config.py          # 独立配置（环境变量加载，不与外部系统冲突）
    ├── voice_clone.py     # 音色克隆管理（上传人声样本 -> 生成/管理 voice_id）
    ├── pipeline.py        # 实时同传核心管线（VAD -> ASR -> MT -> 克隆音色 TTS）
    └── server.py          # 独立 FastAPI + WebSocket 服务（端口 8900）
```

## 快速开始

```bash
cd /Users/mima0000/.gemini/antigravity/scratch/voice-clone-realtime-translator
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# 在 .env 中填入你的 DashScope / MiniMax 等 API Key
uvicorn src.server:app --host 127.0.0.1 --port 8900 --reload
```

## macOS 会议虚拟麦克风联动建议

1. 初期测试可直接安装开源虚拟声卡 **BlackHole 2ch**。
2. 将本项目的音频输出设备指向 `BlackHole 2ch`，在会议软件（Zoom / 腾讯会议 / WhatsApp）中将麦克风设为 `BlackHole 2ch`，即可实现“你说中文 → 对方听见你的克隆音色英文”。
3. 如未来需要与获客系统联动，只需在 `.env` 中配置可选的 `OPTIONAL_WEBHOOK_URL`，通过异步 HTTP 事件推送字幕与会话记录，保持核心服务零耦合。
