"""
api/store.py
Thread-safe, file-backed puzzle store.
Reads/writes api/puzzle_store.json.
Provides O(1) lookup by puzzle id via an in-memory index.
"""

import json
import threading
from pathlib import Path

STORE_PATH = Path("api/puzzle_store.json")


class PuzzleStore:
    """
    Singleton-style store. Multiple PuzzleStore() instances in the same
    process share state via class-level variables.
    """

    _lock:    threading.Lock       = threading.Lock()
    _puzzles: dict[str, dict]      = {}   # id → puzzle dict
    _loaded:  bool                 = False

    def __init__(self):
        if not PuzzleStore._loaded:
            self._load()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        with PuzzleStore._lock:
            if PuzzleStore._loaded:
                return
            STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            if STORE_PATH.exists():
                with open(STORE_PATH) as f:
                    raw = json.load(f)
                PuzzleStore._puzzles = {p["id"]: p for p in raw}
            PuzzleStore._loaded = True

    def _flush(self) -> None:
        """Write current state to disk."""
        with open(STORE_PATH, "w") as f:
            json.dump(list(PuzzleStore._puzzles.values()), f, indent=2)

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, puzzle: dict) -> None:
        """Persist a puzzle. Silently overwrites if id already exists."""
        with PuzzleStore._lock:
            PuzzleStore._puzzles[puzzle["id"]] = puzzle
            self._flush()

    def get(self, puzzle_id: str) -> dict | None:
        return PuzzleStore._puzzles.get(puzzle_id)

    def all(self) -> list[dict]:
        return list(PuzzleStore._puzzles.values())

    def count(self) -> int:
        return len(PuzzleStore._puzzles)

    def delete(self, puzzle_id: str) -> bool:
        """Remove a puzzle by id. Returns True if it existed."""
        with PuzzleStore._lock:
            if puzzle_id in PuzzleStore._puzzles:
                del PuzzleStore._puzzles[puzzle_id]
                self._flush()
                return True
        return False