from __future__ import annotations

import time
from typing import Any

from src.metrics.fps_counter import FPSCounter
from src.metrics.latency_tracker import LatencyTracker
from src.metrics.memory_monitor import MemoryMonitor
from src.metrics.system_monitor import SystemMonitor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsAggregator:
    def __init__(self, enable_gpu: bool = False, system_sampling_interval: float = 1.0) -> None:
        self._fps_counter = FPSCounter()
        self._latency_tracker = LatencyTracker()
        self._memory_monitor = MemoryMonitor(enable_gpu=enable_gpu)
        self._system_monitor = SystemMonitor(sampling_interval=system_sampling_interval)
        self._custom_metrics: dict[str, list[float]] = {}
        self._counters: dict[str, int] = {}
        self._start_time = time.time()

    @property
    def fps_counter(self) -> FPSCounter:
        return self._fps_counter

    @property
    def latency_tracker(self) -> LatencyTracker:
        return self._latency_tracker

    @property
    def memory_monitor(self) -> MemoryMonitor:
        return self._memory_monitor

    @property
    def system_monitor(self) -> SystemMonitor:
        return self._system_monitor

    def tick(self) -> float:
        fps = self._fps_counter.tick()
        self._system_monitor.sample()
        return fps

    def start_stage(self, stage: str) -> None:
        self._latency_tracker.start(stage)

    def stop_stage(self, stage: str) -> float:
        return self._latency_tracker.stop(stage)

    def record_metric(self, name: str, value: float) -> None:
        if name not in self._custom_metrics:
            self._custom_metrics[name] = []
        self._custom_metrics[name].append(value)
        if len(self._custom_metrics[name]) > 10000:
            self._custom_metrics[name] = self._custom_metrics[name][-10000:]

    def increment_counter(self, name: str, amount: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + amount

    def get_counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    def get_snapshot(self) -> dict[str, Any]:
        return {
            "fps": self._fps_counter.get_statistics(),
            "latency": self._latency_tracker.get_all_stats(),
            "pipeline_breakdown": self._latency_tracker.get_pipeline_breakdown(),
            "memory": self._memory_monitor.get_all_memory(),
            "system": {
                "cpu": self._system_monitor.get_cpu_stats(),
                "memory": self._system_monitor.get_memory_stats(),
                "info": self._system_monitor.get_system_info(),
            },
            "counters": dict(self._counters),
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

    def get_summary(self) -> dict[str, Any]:
        fps_stats = self._fps_counter.get_statistics()
        latency = self._latency_tracker.get_all_stats()

        return {
            "fps": fps_stats.get("current_fps", 0),
            "avg_fps": fps_stats.get("average_fps", 0),
            "total_latency_ms": round(self._latency_tracker.get_total_latency_ms(), 1),
            "total_frames": fps_stats.get("total_frames", 0),
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

    def log_summary(self) -> None:
        summary = self.get_summary()
        logger.info("metrics_summary", **summary)

    def reset(self) -> None:
        self._fps_counter.reset()
        self._latency_tracker.reset()
        self._system_monitor.reset()
        self._custom_metrics.clear()
        self._counters.clear()
        self._start_time = time.time()
