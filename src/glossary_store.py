"""Persistent domain glossary store for custom terminology translations."""

from __future__ import annotations

import json
from pathlib import Path


class GlossaryStore:
    """Stores term -> translated_term mappings in a local JSON file."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.path = Path(data_dir) / "glossary.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save({})

    def _load(self) -> dict[str, str]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict[str, str]) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_terms(self) -> dict[str, str]:
        return self._load()

    def upsert_term(self, source_term: str, target_term: str) -> dict[str, str]:
        data = self._load()
        data[source_term.strip()] = target_term.strip()
        self._save(data)
        return data

    def delete_term(self, source_term: str) -> dict[str, str]:
        data = self._load()
        data.pop(source_term, None)
        self._save(data)
        return data
