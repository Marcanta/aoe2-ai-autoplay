"""Region cache: reuse a detection while the pixels it was read from are unchanged.

OCR is deterministic: the same crop, read with the same engine and settings,
gives the same text. Between two observations a second apart most HUD regions
(stone, gold, age, population, the selected unit's name...) do not change, so
re-running Tesseract on them only costs time. The cache keeps, per *slot* (one
detection, e.g. ``wood`` or ``tc_progress``), the fingerprint of the pixels it
was computed from and the result:

* identical pixels (same fingerprint) return the stored result **unchanged**:
  same value, status, confidence, raw text and the timestamp of the frame the
  recognition actually ran on. A cached UNKNOWN stays UNKNOWN; nothing becomes
  VALID that recognition did not make VALID.
* any changed pixel misses and the detection runs again.
* a change of resolution, region set, OCR engine/settings or detector settings
  (the *configuration key*) empties the whole cache.

The fingerprint is a 128-bit BLAKE2b digest of the crop's shape and raw bytes,
so "identical" means bit-identical: a one-level change in one pixel misses.

Because a reused detection keeps its original timestamp, a value that stays on
screen unchanged and then becomes unreadable goes UNKNOWN sooner than without
the cache (``StateTracker`` measures staleness from the detection timestamp).
That is the conservative direction: the cache never extends how long a value
is trusted.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Callable, Hashable, TypeVar

import numpy as np

T = TypeVar("T")


def fingerprint(crop: np.ndarray) -> bytes:
    """Digest of a crop's shape, dtype and pixels."""
    h = hashlib.blake2b(digest_size=16)
    h.update(repr((crop.shape, crop.dtype.str)).encode())
    h.update(np.ascontiguousarray(crop).tobytes())
    return h.digest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    invalidations: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class RegionCache:
    """Last result per slot, keyed by the fingerprints of the pixels it depends on."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[Hashable, Any]] = {}
        self._config: Hashable | None = None
        self.stats = CacheStats()

    def configure(self, config_key: Hashable) -> None:
        """Set the configuration the cached results belong to; a different one clears the cache."""
        with self._lock:
            if config_key != self._config:
                if self._entries:
                    self.stats.invalidations += 1
                self._entries.clear()
                self._config = config_key

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def key(self, crops: tuple[np.ndarray, ...], *params: Hashable) -> tuple:
        """Cache key for a result computed from ``crops`` with ``params`` (thresholds, word lists...)."""
        return (tuple(fingerprint(c) for c in crops), params)

    def get(self, slot: str, key: Hashable) -> Any | None:
        with self._lock:
            entry = self._entries.get(slot)
            if entry is not None and entry[0] == key:
                self.stats.hits += 1
                return entry[1]
            self.stats.misses += 1
            return None

    def put(self, slot: str, key: Hashable, value: Any) -> None:
        if value is None:
            raise ValueError("cannot cache None (it means 'miss')")
        with self._lock:
            self._entries[slot] = (key, value)

    def lookup_or_compute(self, slot: str, key: Hashable, compute: Callable[[], T]) -> T:
        cached = self.get(slot, key)
        if cached is not None:
            return cached
        value = compute()
        self.put(slot, key, value)
        return value


def cached(cache: RegionCache | None, slot: str, crops: tuple[np.ndarray, ...], params: tuple, compute: Callable[[], T]) -> T:
    """``compute()`` through ``cache`` when there is one."""
    if cache is None:
        return compute()
    return cache.lookup_or_compute(slot, cache.key(crops, *params), compute)
