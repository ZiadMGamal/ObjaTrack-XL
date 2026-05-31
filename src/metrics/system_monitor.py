from __future__ import annotations

import os
import platform
import time
from typing import Any

import psutil


class SystemMonitor:

    def __init__(self, sampling_interval: float = 1.0) -> None:
        self._sampling_interval = sampling_interval
        self._process = psutil.Process(os.getpid())
        self._last_sample_time: float = 0.0
        self._cpu_history: list[float] = []
        self._memory_history: list[float] = []
        self._max_history: int = 3600

    def sample(self) -> dict[str, Any]:
        now = time.time()
        if now - self._last_sample_time < self._sampling_interval:
            return {}

        self._last_sample_time = now
        cpu_percent = self._process.cpu_percent()
        memory_percent = self._process.memory_percent()

        self._cpu_history.append(cpu_percent)
        self._memory_history.append(memory_percent)

        if len(self._cpu_history) > self._max_history:
            self._cpu_history = self._cpu_history[-self._max_history:]
            self._memory_history = self._memory_history[-self._max_history:]

        return {
            "cpu_percent": round(cpu_percent, 1),
            "memory_percent": round(memory_percent, 1),
            "threads": self._process.num_threads(),
            "timestamp": now,
        }

    def get_system_info(self) -> dict[str, Any]:
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
            "python_version": platform.python_version(),
            "pid": os.getpid(),
        }

    def get_process_info(self) -> dict[str, Any]:
        try:
            mem_info = self._process.memory_info()
            return {
                "pid": self._process.pid,
                "cpu_percent": round(self._process.cpu_percent(), 1),
                "memory_rss_mb": round(mem_info.rss / (1024 * 1024), 1),
                "memory_percent": round(self._process.memory_percent(), 1),
                "threads": self._process.num_threads(),
                "status": self._process.status(),
                "create_time": self._process.create_time(),
            }
        except psutil.NoSuchProcess:
            return {}

    def get_cpu_stats(self) -> dict[str, float]:
        if not self._cpu_history:
            return {"avg": 0.0, "max": 0.0, "min": 0.0, "current": 0.0}
        return {
            "avg": round(sum(self._cpu_history) / len(self._cpu_history), 1),
            "max": round(max(self._cpu_history), 1),
            "min": round(min(self._cpu_history), 1),
            "current": round(self._cpu_history[-1], 1),
            "samples": len(self._cpu_history),
        }

    def get_memory_stats(self) -> dict[str, float]:
        if not self._memory_history:
            return {"avg": 0.0, "max": 0.0, "min": 0.0, "current": 0.0}
        return {
            "avg": round(sum(self._memory_history) / len(self._memory_history), 1),
            "max": round(max(self._memory_history), 1),
            "min": round(min(self._memory_history), 1),
            "current": round(self._memory_history[-1], 1),
            "samples": len(self._memory_history),
        }

    def reset(self) -> None:
        self._cpu_history.clear()
        self._memory_history.clear()
        self._last_sample_time = 0.0
