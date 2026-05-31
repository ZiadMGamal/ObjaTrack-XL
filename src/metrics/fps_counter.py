from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    def __init__(self, window_size: int = 60) -> None:
        self._window_size = window_size
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._total_frames: int = 0
        self._start_time: float = 0.0

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def start(self) -> None:
        self._start_time = time.time()

    def tick(self) -> float:
        now = time.time()
        self._timestamps.append(now)
        self._total_frames += 1

        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(self._timestamps) - 1) / elapsed

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    @property
    def average_fps(self) -> float:
        if self._total_frames == 0 or self._start_time == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._total_frames / elapsed

    def reset(self) -> None:
        self._timestamps.clear()
        self._total_frames = 0
        self._start_time = time.time()

    def get_statistics(self) -> dict[str, float]:
        return {
            "current_fps": round(self.fps, 1),
            "average_fps": round(self.average_fps, 1),
            "total_frames": self._total_frames,
            "elapsed_seconds": round(time.time() - self._start_time, 1) if self._start_time > 0 else 0.0,
        }
