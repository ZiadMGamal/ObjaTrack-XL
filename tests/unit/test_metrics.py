from __future__ import annotations

import time
import pytest

from src.metrics.fps_counter import FPSCounter
from src.metrics.latency_tracker import LatencyTracker
from src.metrics.metrics_aggregator import MetricsAggregator


class TestFPSCounter:

    def test_initialization(self) -> None:
        counter = FPSCounter()
        assert counter.total_frames == 0
        assert counter.fps == 0.0

    def test_tick(self) -> None:
        counter = FPSCounter()
        for _ in range(10):
            counter.tick()
        assert counter.total_frames == 10

    def test_reset(self) -> None:
        counter = FPSCounter()
        counter.tick()
        counter.tick()
        counter.reset()
        assert counter.total_frames == 0

    def test_statistics(self) -> None:
        counter = FPSCounter()
        counter.start()
        for _ in range(5):
            counter.tick()
        stats = counter.get_statistics()
        assert "current_fps" in stats
        assert "average_fps" in stats
        assert "total_frames" in stats
        assert stats["total_frames"] == 5


class TestLatencyTracker:

    def test_create_stage(self) -> None:
        tracker = LatencyTracker()
        tracker.create_stage("detection")
        stats = tracker.get_stage_stats("detection")
        assert isinstance(stats, dict)

    def test_start_stop(self) -> None:
        tracker = LatencyTracker()
        tracker.start("detection")
        time.sleep(0.01)
        latency = tracker.stop("detection")
        assert latency > 0

    def test_stop_without_start(self) -> None:
        tracker = LatencyTracker()
        latency = tracker.stop("nonexistent")
        assert latency == 0.0

    def test_total_latency(self) -> None:
        tracker = LatencyTracker()
        tracker.start("a")
        time.sleep(0.01)
        tracker.stop("a")
        tracker.start("b")
        time.sleep(0.01)
        tracker.stop("b")
        total = tracker.get_total_latency_ms()
        assert total > 0

    def test_reset(self) -> None:
        tracker = LatencyTracker()
        tracker.start("test")
        tracker.stop("test")
        tracker.reset()
        assert tracker.get_total_latency_ms() == 0.0


class TestMetricsAggregator:

    def test_initialization(self) -> None:
        agg = MetricsAggregator()
        assert agg.fps_counter is not None
        assert agg.latency_tracker is not None
        assert agg.memory_monitor is not None

    def test_tick(self) -> None:
        agg = MetricsAggregator()
        agg.fps_counter.start()
        fps = agg.tick()
        assert isinstance(fps, float)

    def test_stage_timing(self) -> None:
        agg = MetricsAggregator()
        agg.start_stage("test")
        time.sleep(0.01)
        latency = agg.stop_stage("test")
        assert latency > 0

    def test_counters(self) -> None:
        agg = MetricsAggregator()
        agg.increment_counter("detections", 5)
        agg.increment_counter("detections", 3)
        assert agg.get_counter("detections") == 8

    def test_custom_metrics(self) -> None:
        agg = MetricsAggregator()
        agg.record_metric("confidence", 0.85)
        agg.record_metric("confidence", 0.92)

    def test_snapshot(self) -> None:
        agg = MetricsAggregator()
        agg.fps_counter.start()
        agg.tick()
        snapshot = agg.get_snapshot()
        assert "fps" in snapshot
        assert "latency" in snapshot
        assert "memory" in snapshot
        assert "counters" in snapshot

    def test_summary(self) -> None:
        agg = MetricsAggregator()
        agg.fps_counter.start()
        summary = agg.get_summary()
        assert "fps" in summary
        assert "total_frames" in summary

    def test_reset(self) -> None:
        agg = MetricsAggregator()
        agg.increment_counter("test", 10)
        agg.reset()
        assert agg.get_counter("test") == 0
