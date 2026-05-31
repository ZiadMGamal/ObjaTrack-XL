from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.tracking.track import Track
from src.utils.logger import get_logger
from src.utils.math_utils import MathUtils

logger = get_logger(__name__)


class ObjectCounter:
    def __init__(
        self,
        line_start: tuple[int, int] = (0, 360),
        line_end: tuple[int, int] = (1280, 360),
        direction: str = "horizontal",
        count_classes: list[str] | None = None,
    ) -> None:
        self._line_start = line_start
        self._line_end = line_end
        self._direction = direction
        self._count_classes = count_classes
        self._count_in: int = 0
        self._count_out: int = 0
        self._counted_ids: set[int] = set()
        self._previous_positions: dict[int, tuple[float, float]] = {}
        self._class_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})

    @property
    def count_in(self) -> int:
        return self._count_in

    @property
    def count_out(self) -> int:
        return self._count_out

    @property
    def total_count(self) -> int:
        return self._count_in + self._count_out

    @property
    def line_start(self) -> tuple[int, int]:
        return self._line_start

    @property
    def line_end(self) -> tuple[int, int]:
        return self._line_end

    def update(self, tracks: list[Track]) -> list[dict[str, Any]]:
        events = []

        for track in tracks:
            if self._count_classes and track.class_name not in self._count_classes:
                continue

            center = track.center
            prev = self._previous_positions.get(track.track_id)

            if prev is not None and track.track_id not in self._counted_ids:
                crossed = self._check_line_crossing(prev, center)
                if crossed != 0:
                    self._counted_ids.add(track.track_id)
                    if crossed > 0:
                        self._count_in += 1
                        self._class_counts[track.class_name]["in"] += 1
                        direction = "in"
                    else:
                        self._count_out += 1
                        self._class_counts[track.class_name]["out"] += 1
                        direction = "out"

                    event = {
                        "type": "line_crossing",
                        "track_id": track.track_id,
                        "class_name": track.class_name,
                        "direction": direction,
                        "position": center,
                        "count_in": self._count_in,
                        "count_out": self._count_out,
                    }
                    events.append(event)
                    logger.info("line_crossing", **event)

            self._previous_positions[track.track_id] = center

        active_ids = {t.track_id for t in tracks}
        stale = [tid for tid in self._previous_positions if tid not in active_ids]
        for tid in stale:
            del self._previous_positions[tid]

        return events

    def _check_line_crossing(
        self,
        prev: tuple[float, float],
        current: tuple[float, float],
    ) -> int:
        x1, y1 = self._line_start
        x2, y2 = self._line_end

        d_prev = (x2 - x1) * (prev[1] - y1) - (y2 - y1) * (prev[0] - x1)
        d_curr = (x2 - x1) * (current[1] - y1) - (y2 - y1) * (current[0] - x1)

        if d_prev * d_curr < 0:
            intersection = MathUtils.line_intersection(prev, current, (float(x1), float(y1)), (float(x2), float(y2)))
            if intersection is not None:
                return 1 if d_curr > 0 else -1

        return 0

    def get_statistics(self) -> dict[str, Any]:
        return {
            "count_in": self._count_in,
            "count_out": self._count_out,
            "total": self.total_count,
            "class_counts": dict(self._class_counts),
            "unique_counted": len(self._counted_ids),
        }

    def reset(self) -> None:
        self._count_in = 0
        self._count_out = 0
        self._counted_ids.clear()
        self._previous_positions.clear()
        self._class_counts.clear()
