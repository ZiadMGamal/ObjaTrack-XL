from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.tracking.track import Track
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SpeedEstimator:
    def __init__(
        self,
        pixels_per_meter: float = 8.0,
        fps: float = 30.0,
        smoothing_window: int = 5,
        min_track_length: int = 3,
        speed_unit: str = "km/h",
    ) -> None:
        self._pixels_per_meter = pixels_per_meter
        self._fps = fps
        self._smoothing_window = smoothing_window
        self._min_track_length = min_track_length
        self._speed_unit = speed_unit
        self._track_speeds: dict[int, list[float]] = defaultdict(list)
        self._current_speeds: dict[int, float] = {}

    @property
    def pixels_per_meter(self) -> float:
        return self._pixels_per_meter

    @pixels_per_meter.setter
    def pixels_per_meter(self, value: float) -> None:
        self._pixels_per_meter = max(0.1, value)

    def update(self, tracks: list[Track]) -> dict[int, float]:
        speeds = {}

        for track in tracks:
            if track.track_length < self._min_track_length:
                continue

            trajectory = track.trajectory
            if len(trajectory) < 2:
                continue

            p1 = trajectory[-2]
            p2 = trajectory[-1]
            pixel_dist = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            meter_dist = pixel_dist / self._pixels_per_meter
            time_interval = 1.0 / self._fps
            speed_mps = meter_dist / time_interval

            if self._speed_unit == "km/h":
                speed = speed_mps * 3.6
            elif self._speed_unit == "mph":
                speed = speed_mps * 2.237
            else:
                speed = speed_mps

            self._track_speeds[track.track_id].append(speed)
            if len(self._track_speeds[track.track_id]) > self._smoothing_window:
                self._track_speeds[track.track_id] = self._track_speeds[track.track_id][-self._smoothing_window :]

            smoothed = sum(self._track_speeds[track.track_id]) / len(self._track_speeds[track.track_id])
            self._current_speeds[track.track_id] = smoothed
            speeds[track.track_id] = round(smoothed, 1)

        return speeds

    def get_speed(self, track_id: int) -> float:
        return self._current_speeds.get(track_id, 0.0)

    def get_all_speeds(self) -> dict[int, float]:
        return dict(self._current_speeds)

    def get_statistics(self) -> dict[str, Any]:
        if not self._current_speeds:
            return {"avg_speed": 0.0, "max_speed": 0.0, "min_speed": 0.0, "num_tracked": 0}

        values = list(self._current_speeds.values())
        return {
            "avg_speed": round(sum(values) / len(values), 1),
            "max_speed": round(max(values), 1),
            "min_speed": round(min(values), 1),
            "num_tracked": len(values),
            "unit": self._speed_unit,
        }

    def reset(self) -> None:
        self._track_speeds.clear()
        self._current_speeds.clear()
