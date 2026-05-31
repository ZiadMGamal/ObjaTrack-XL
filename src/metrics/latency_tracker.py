from __future__ import annotations

import time

from src.utils.timer import AccumulatingTimer


class LatencyTracker:
    def __init__(self) -> None:
        self._timers: dict[str, AccumulatingTimer] = {}
        self._current_start: dict[str, float] = {}

    def create_stage(self, name: str) -> None:
        if name not in self._timers:
            self._timers[name] = AccumulatingTimer(name=name)

    def start(self, stage: str) -> None:
        self.create_stage(stage)
        self._current_start[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        if stage not in self._current_start:
            return 0.0

        elapsed = time.perf_counter() - self._current_start[stage]
        self._timers[stage].record(elapsed)
        del self._current_start[stage]
        return elapsed * 1000.0

    def measure(self, stage: str) -> AccumulatingTimer:
        self.create_stage(stage)
        return self._timers[stage].measure()

    def get_stage_stats(self, stage: str) -> dict[str, float]:
        if stage not in self._timers:
            return {}
        return self._timers[stage].get_statistics()

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        return {name: timer.get_statistics() for name, timer in self._timers.items()}

    def get_total_latency_ms(self) -> float:
        total = 0.0
        for timer in self._timers.values():
            if timer.count > 0:
                total += timer.average_time_ms
        return total

    def get_pipeline_breakdown(self) -> dict[str, float]:
        total = self.get_total_latency_ms()
        if total == 0:
            return {}

        breakdown = {}
        for name, timer in self._timers.items():
            if timer.count > 0:
                avg = timer.average_time_ms
                breakdown[name] = {
                    "avg_ms": round(avg, 3),
                    "percentage": round((avg / total) * 100, 1),
                }
        return breakdown

    def reset(self) -> None:
        for timer in self._timers.values():
            timer.reset()
        self._current_start.clear()

    def reset_stage(self, stage: str) -> None:
        if stage in self._timers:
            self._timers[stage].reset()
