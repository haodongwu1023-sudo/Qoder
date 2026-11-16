"""Persistent translation session history store."""

from __future__ import annotations

import json
import os
import tempfile
from functools import wraps
from threading import Lock
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _locked(method):
    @wraps(method)
    def inner(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return inner


class HistoryStore:
    """Stores completed translation utterances grouped by session in data/history.json."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self._lock = Lock()
        self.path = Path(data_dir) / "history.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save([])

    def _load(self) -> list[dict[str, Any]]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, sessions: list[dict[str, Any]]) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as output:
            tmp = Path(output.name)
            try:
                json.dump(sessions, output, ensure_ascii=False, indent=2)
                output.close()
                os.replace(tmp, self.path)
            finally:
                tmp.unlink(missing_ok=True)

    @_locked
    def start_session(self, source_lang: str, target_lang: str, voice_id: str) -> str:
        sessions = self._load()
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        sessions.insert(
            0,
            {
                "session_id": session_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "source_lang": source_lang,
                "target_lang": target_lang,
                "voice_id": voice_id,
                "items": [],
            },
        )
        # Keep latest 50 sessions
        self._save(sessions[:50])
        return session_id

    @_locked
    def append_item(self, session_id: str, source_text: str, target_text: str) -> None:
        sessions = self._load()
        for sess in sessions:
            if sess["session_id"] == session_id:
                sess["items"].append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": source_text,
                        "target": target_text,
                    }
                )
                break
        self._save(sessions)

    @_locked
    def update_last_source(self, session_id: str, placeholder_source: str, actual_source: str) -> None:
        """Replace a placeholder source text on the most recent item if ASR final arrives slightly after MT final."""
        sessions = self._load()
        for sess in sessions:
            if sess["session_id"] == session_id and sess.get("items"):
                last_item = sess["items"][-1]
                if last_item.get("source") == placeholder_source:
                    last_item["source"] = actual_source
                    self._save(sessions)
                break

    @_locked
    def list_sessions(self) -> list[dict[str, Any]]:
        return self._load()

    @_locked
    def clear(self) -> None:
        self._save([])
