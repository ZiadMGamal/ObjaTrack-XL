from __future__ import annotations

import numpy as np

from src.analytics.counter import ObjectCounter
from src.analytics.dwell_time import DwellTimeAnalyzer
from src.analytics.speed_estimator import SpeedEstimator
from src.tracking.track import Track, TrackState


def _make_track(track_id: int, center: tuple[float, float], class_name: str = "person") -> Track:
    cx, cy = center
    w, h = 50, 100
    return Track(
        track_id=track_id,
        box=np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]),
        score=0.9,
        class_id=0,
        class_name=class_name,
        state=TrackState.CONFIRMED,
        trajectory=[(cx, cy)],
    )


class TestObjectCounter:
    def test_initialization(self) -> None:
        counter = ObjectCounter(
            line_start=(0, 360),
            line_end=(1280, 360),
        )
        assert counter.count_in == 0
        assert counter.count_out == 0
        assert counter.total_count == 0

    def test_no_crossing(self) -> None:
        counter = ObjectCounter(
            line_start=(0, 360),
            line_end=(1280, 360),
        )
        track = _make_track(1, (100, 200))
        events = counter.update([track])
        assert len(events) == 0

    def test_crossing_detection(self) -> None:
        counter = ObjectCounter(
            line_start=(0, 360),
            line_end=(1280, 360),
        )
        track1 = _make_track(1, (100, 300))
        counter.update([track1])

        track2 = _make_track(1, (100, 420))
        events = counter.update([track2])
        assert counter.total_count == 1

    def test_statistics(self) -> None:
        counter = ObjectCounter()
        stats = counter.get_statistics()
        assert "count_in" in stats
        assert "count_out" in stats
        assert "total" in stats

    def test_reset(self) -> None:
        counter = ObjectCounter()
        counter._count_in = 5
        counter._count_out = 3
        counter.reset()
        assert counter.count_in == 0
        assert counter.count_out == 0


class TestSpeedEstimator:
    def test_initialization(self) -> None:
        estimator = SpeedEstimator(pixels_per_meter=10.0, fps=30.0)
        assert estimator.pixels_per_meter == 10.0

    def test_no_speed_short_track(self) -> None:
        estimator = SpeedEstimator()
        track = _make_track(1, (100, 100))
        speeds = estimator.update([track])
        assert len(speeds) == 0

    def test_speed_calculation(self) -> None:
        estimator = SpeedEstimator(pixels_per_meter=10.0, fps=30.0)

        track = _make_track(1, (100, 100))
        track.trajectory = [(100, 100), (110, 100), (120, 100)]

        speeds = estimator.update([track])
        assert 1 in speeds
        assert speeds[1] > 0

    def test_statistics(self) -> None:
        estimator = SpeedEstimator()
        stats = estimator.get_statistics()
        assert "avg_speed" in stats
        assert "max_speed" in stats

    def test_reset(self) -> None:
        estimator = SpeedEstimator()
        estimator._current_speeds[1] = 50.0
        estimator.reset()
        assert len(estimator.get_all_speeds()) == 0


class TestDwellTimeAnalyzer:
    def test_initialization(self) -> None:
        analyzer = DwellTimeAnalyzer(threshold_seconds=10.0)
        assert analyzer.threshold == 10.0

    def test_track_entry(self) -> None:
        analyzer = DwellTimeAnalyzer()
        track = _make_track(1, (100, 100))
        events = analyzer.update([track])
        dwell = analyzer.get_dwell_time(1)
        assert dwell >= 0

    def test_statistics(self) -> None:
        analyzer = DwellTimeAnalyzer()
        stats = analyzer.get_statistics()
        assert "active_count" in stats

    def test_reset(self) -> None:
        analyzer = DwellTimeAnalyzer()
        track = _make_track(1, (100, 100))
        analyzer.update([track])
        analyzer.reset()
        assert analyzer.get_dwell_time(1) == 0.0
