# EchoBridge Web · 原声跨语实时同传（网页比赛版）

这是一个**完全独立于桌面端**的纯网页版实时语音同传与声音复刻项目，继承了桌面版已经调优好的核心音频与实时参数，并针对浏览器场景与 **Qoder Sites** 部署做了彻底解耦。

## 继承自原 App 的核心调优参数

| 模块 | 参数 | 说明 |
| :--- | :--- | :--- |
| 端到端实时同传 | `qwen3.5-livetranslate-flash-realtime` | `24kHz PCM16` 流式译音直出 + 实时原声复刻 (`enable_voice_clone=True`, `frequency="always"`) |
| 断句灵敏度 | `VAD_SILENCE_MS=400`, `threshold=0.45` | 相比默认 750ms 显著降低一来一回对话等待延迟 |
| 长连接保活 | `KEEPALIVE_INTERVAL_MS=100`, `KEEPALIVE_IDLE_MS=150` | 静音帧保活保持服务端时钟连续，防止停顿后首词卡顿 |
| 前景语音门控 | `ForegroundSpeechGate` (`base_open_dbfs=-42.0`, 瞬态脉冲剔除) | 配合浏览器 WebRTC AEC (`echoCancellation` + `noiseSuppression`) 过滤敲击与背景杂音 |
| 自定义复刻音色 | MiniMax `speech-2.8-turbo` + `wss://api.minimaxi.com/ws/v1/t2a_v2` | 浏览器 10 秒录音编码 16kHz WAV 上传复刻，存在访客浏览器本地 `localStorage` |

## 项目结构

```text
echobridge-web/
├── frontend/                 # 纯静态前端（可直接用 Qoder 发布到 Qoder Sites）
│   ├── index.html            # 面对面同传 / 双人跨语房间 / 10 秒复刻 / 免开麦试听样例
│   ├── audio-worklet.js      # AudioWorklet 线程实时降采样 16kHz PCM16 (40ms/帧)
│   ├── config.js             # 后端服务地址配置（部署 Qoder Sites 时填入后端 HTTPS 域名）
│   └── samples/              # 内置中英文样例 WAV，评委无麦克风也能 10 秒测通
├── backend/                  # 无声卡驱动依赖的轻量多用户 WebSocket 网关
│   ├── app/
│   │   ├── server.py         # FastAPI + CORS + WebSocket (/ws/web-translate, /ws/room/{id})
│   │   ├── web_session.py    # 按连接隔离会话 + 房间双语语音/字幕转发 Hub
│   │   ├── livetranslate_client.py
│   │   ├── minimax_client.py
│   │   ├── audio_gate.py
│   │   └── config.py
│   └── requirements.txt
├── Dockerfile                # 云端容器一键部署实时后端
├── export_qoder_sites.sh     # 导出 Qoder Sites 静态站点包到 qoder_sites_bundle/
└── run.sh                    # 本机一键调试启动 (http://127.0.0.1:8910)
```

## 本地试跑

```bash
cd echobridge-web
./run.sh
```
浏览器打开 `http://127.0.0.1:8910` 即可体验。

## 部署到 Qoder Sites + 云端后端

1. **部署轻量 Python 实时后端（支持 `wss://` 的容器/ECS）**：
   ```bash
   docker build -t echobridge-web .
   docker run -d -p 8080:8080 \
     -e DASHSCOPE_API_KEY="你的Key" \
     -e WEB_DEMO_MAX_SECONDS=180 \
     -e WEB_ALLOWED_ORIGINS="https://你的站点.qoder.zone" \
     echobridge-web
   ```
2. **导出前端并交给 Qoder 部署到 Qoder Sites**：
   ```bash
   ./export_qoder_sites.sh "https://你的后端HTTPS域名"
   ```
   打开 Qoder，将 `qoder_sites_bundle/`（或 `frontend/`）作为站点目录通过 Qoder Sites 发布，即可获得公开 `https://*.qoder.zone` 比赛访问链接。
