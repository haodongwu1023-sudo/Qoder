"""Local voice library registry for cloned voices and recorded audio samples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ClonedVoiceEntry:
    voice_id: str
    name: str
    sample_path: str
    file_id: str = ""
    created_at: str = ""
    is_default: bool = False


class VoiceLibrary:
    """Manages persisted cloned voice profiles in a local JSON file."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.samples_dir = self.data_dir / "samples"
        self.registry_path = self.data_dir / "voices.json"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._save([])

    def _load(self) -> list[ClonedVoiceEntry]:
        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        return [ClonedVoiceEntry(**item) for item in raw]

    def _save(self, entries: list[ClonedVoiceEntry]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(e) for e in entries]
        self.registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_voices(self) -> list[dict]:
        return [asdict(e) for e in self._load()]

    def add_voice(self, voice_id: str, name: str, sample_path: str, file_id: str = "") -> dict:
        entries = [e for e in self._load() if e.voice_id != voice_id]
        is_first = len(entries) == 0
        entry = ClonedVoiceEntry(
            voice_id=voice_id,
            name=name or voice_id,
            sample_path=sample_path,
            file_id=file_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            is_default=is_first,
        )
        entries.append(entry)
        self._save(entries)
        return asdict(entry)

    def set_default(self, voice_id: str) -> dict | None:
        entries = self._load()
        target = None
        for e in entries:
            e.is_default = e.voice_id == voice_id
            if e.is_default:
                target = asdict(e)
        self._save(entries)
        return target

    def get_default_voice_id(self) -> str:
        for e in self._load():
            if e.is_default:
                return e.voice_id
        entries = self._load()
        return entries[0].voice_id if entries else ""

    def delete_voice(self, voice_id: str) -> bool:
        entries = self._load()
        remaining = [e for e in entries if e.voice_id != voice_id]
        if len(remaining) == len(entries):
            return False
        if remaining and not any(e.is_default for e in remaining):
            remaining[0].is_default = True
        self._save(remaining)
        return True
