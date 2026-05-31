from __future__ import annotations

import time
from typing import Any
from collections import defaultdict

from src.tracking.track import Track
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DwellTimeAnalyzer:

    def __init__(
        self,
        threshold_seconds: float = 10.0,
        alert_on_threshold: bool = True,
    ) -> None:
        self._threshold = threshold_seconds
        self._alert_on_threshold = alert_on_threshold
        self._entry_times: dict[int, float] = {}
        self._dwell_times: dict[int, float] = {}
        self._alerted_ids: set[int] = set()
        self._class_dwell: dict[str, list[float]] = defaultdict(list)

    @property
    def threshold(self) -> float:
        return self._threshold

    def update(self, tracks: list[Track]) -> list[dict[str, Any]]:
        events = []
        current_time = time.time()
        active_ids = set()

        for track in tracks:
            active_ids.add(track.track_id)

            if track.track_id not in self._entry_times:
                self._entry_times[track.track_id] = current_time

            dwell = current_time - self._entry_times[track.track_id]
            self._dwell_times[track.track_id] = dwell

            if (self._alert_on_threshold
                and dwell >= self._threshold
                and track.track_id not in self._alerted_ids):
                self._alerted_ids.add(track.track_id)
                event = {
                    "type": "dwell_threshold",
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "dwell_time": round(dwell, 1),
                    "threshold": self._threshold,
                }
                events.append(event)
                logger.info("dwell_threshold_exceeded", **event)

        departed = set(self._entry_times.keys()) - active_ids
        for tid in departed:
            if tid in self._dwell_times:
                final_dwell = self._dwell_times[tid]
                self._class_dwell["all"].append(final_dwell)
            self._entry_times.pop(tid, None)
            self._dwell_times.pop(tid, None)
            self._alerted_ids.discard(tid)

        return events

    def get_dwell_time(self, track_id: int) -> float:
        return self._dwell_times.get(track_id, 0.0)

    def get_all_dwell_times(self) -> dict[int, float]:
        return dict(self._dwell_times)

    def get_statistics(self) -> dict[str, Any]:
        active = list(self._dwell_times.values())
        historical = self._class_dwell.get("all", [])

        return {
            "active_count": len(active),
            "avg_active_dwell": round(sum(active) / len(active), 1) if active else 0.0,
            "max_active_dwell": round(max(active), 1) if active else 0.0,
            "total_historical": len(historical),
            "avg_historical_dwell": round(sum(historical) / len(historical), 1) if historical else 0.0,
            "threshold_alerts": len(self._alerted_ids),
        }

    def reset(self) -> None:
        self._entry_times.clear()
        self._dwell_times.clear()
        self._alerted_ids.clear()
        self._class_dwell.clear()
