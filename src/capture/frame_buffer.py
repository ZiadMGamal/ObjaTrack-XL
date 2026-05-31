from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FrameBuffer:
    def __init__(
        self,
        max_size: int = 128,
        drop_strategy: str = "oldest",
        name: str = "default",
    ) -> None:
        self._max_size = max_size
        self._drop_strategy = drop_strategy
        self._name = name
        self._buffer: deque[FrameEntry] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._total_pushed: int = 0
        self._total_popped: int = 0
        self._total_dropped: int = 0
        self._closed = False

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) >= self._max_size

    @property
    def utilization(self) -> float:
        with self._lock:
            return len(self._buffer) / self._max_size if self._max_size > 0 else 0.0

    def push(self, frame: np.ndarray, metadata: dict[str, Any] | None = None, timeout: float | None = None) -> bool:
        with self._not_full:
            if self._closed:
                return False

            if len(self._buffer) >= self._max_size:
                if self._drop_strategy == "oldest":
                    self._buffer.popleft()
                    self._total_dropped += 1
                elif self._drop_strategy == "newest":
                    self._total_dropped += 1
                    return False
                elif self._drop_strategy == "block":
                    if not self._not_full.wait(timeout=timeout):
                        return False

            entry = FrameEntry(
                frame=frame,
                timestamp=time.time(),
                frame_id=self._total_pushed,
                metadata=metadata or {},
            )
            self._buffer.append(entry)
            self._total_pushed += 1
            self._not_empty.notify()
            return True

    def pop(self, timeout: float | None = None) -> FrameEntry | None:
        with self._not_empty:
            while len(self._buffer) == 0:
                if self._closed:
                    return None
                if not self._not_empty.wait(timeout=timeout):
                    return None

            entry = self._buffer.popleft()
            self._total_popped += 1
            self._not_full.notify()
            return entry

    def peek(self) -> FrameEntry | None:
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[0]

    def peek_latest(self) -> FrameEntry | None:
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1]

    def clear(self) -> int:
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            self._not_full.notify_all()
            return count

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self._name,
                "current_size": len(self._buffer),
                "max_size": self._max_size,
                "utilization": f"{self.utilization:.1%}",
                "total_pushed": self._total_pushed,
                "total_popped": self._total_popped,
                "total_dropped": self._total_dropped,
                "drop_rate": f"{self._total_dropped / max(1, self._total_pushed):.2%}",
            }

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"FrameBuffer(name={self._name!r}, size={self.size}/{self._max_size})"


class FrameEntry:
    __slots__ = ("frame", "timestamp", "frame_id", "metadata")

    def __init__(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_id: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.frame = frame
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.metadata = metadata or {}

    @property
    def shape(self) -> tuple[int, ...]:
        return self.frame.shape

    @property
    def age_ms(self) -> float:
        return (time.time() - self.timestamp) * 1000.0

    def __repr__(self) -> str:
        return f"FrameEntry(id={self.frame_id}, shape={self.shape}, age_ms={self.age_ms:.1f})"
