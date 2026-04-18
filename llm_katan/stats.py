"""
Persistent request counters for LLM Katan.

Tracks total and per-provider request counts across server restarts.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PersistentStats:
    """Global request counters that persist to a JSON file."""

    def __init__(self, path: str | None = None):
        self._path = Path(path) if path else None
        self._total: int = 0
        self._providers: dict[str, int] = {}
        if self._path:
            self._load()

    def _load(self):
        if self._path and self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._total = data.get("total", 0)
                self._providers = data.get("providers", {})
                logger.info("Loaded persistent stats: %d total requests", self._total)
            except Exception:
                logger.warning("Could not load stats from %s, starting fresh", self._path)

    def _save(self):
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self.get(), indent=2) + "\n")

    def record(self, provider: str):
        self._total += 1
        self._providers[provider] = self._providers.get(provider, 0) + 1
        self._save()

    def get(self) -> dict:
        return {
            "total": self._total,
            "providers": dict(self._providers),
        }

    @property
    def total(self) -> int:
        return self._total

    @property
    def providers(self) -> dict[str, int]:
        return dict(self._providers)
