"""Native desktop launcher using pywebview (macOS WKWebView / Windows WebView2)."""

from __future__ import annotations

import socket
import threading
import time

import uvicorn
import webview

from src.config import settings
from src.server import app


def _find_free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class DesktopBridge:
    """Exposed to JS as window.pywebview.api for managing the floating subtitle overlay."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.overlay_window: webview.Window | None = None

    def toggle_overlay(self) -> bool:
        if self.overlay_window is None:
            self.overlay_window = webview.create_window(
                "悬浮双语字幕",
                f"{self.base_url}/overlay",
                width=680,
                height=220,
                frameless=True,
                on_top=True,
                easy_drag=True,
                resizable=True,
                js_api=self,
            )
            return True
        self.overlay_window.show()
        return True

    def set_overlay_height(self, height: int) -> bool:
        if self.overlay_window is not None:
            try:
                self.overlay_window.resize(680, max(120, min(320, int(height))))
            except Exception:
                return False
        return True

    def hide_overlay(self) -> bool:
        if self.overlay_window is not None:
            self.overlay_window.hide()
        return True


def _run_server(port: int) -> None:
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", loop="asyncio")


def main() -> None:
    port = _find_free_port(settings.app_port)
    base_url = f"http://127.0.0.1:{port}"

    server_thread = threading.Thread(target=_run_server, args=(port,), daemon=True)
    server_thread.start()

    # Wait briefly for uvicorn to bind
    for _ in range(40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.1)

    bridge = DesktopBridge(base_url=base_url)
    webview.create_window(
        "VoiceClone Live — 实时同传与音色克隆",
        f"{base_url}/?v={int(time.time())}",
        width=1060,
        height=720,
        min_size=(860, 580),
        js_api=bridge,
    )
    webview.start()


if __name__ == "__main__":
    main()
