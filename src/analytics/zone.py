from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.tracking.track import Track
from src.utils.logger import get_logger
from src.utils.math_utils import MathUtils

logger = get_logger(__name__)


class Zone:
    def __init__(self, zone_id: str, points: list[tuple[int, int]], name: str = "") -> None:
        self.zone_id = zone_id
        self.points = points
        self.name = name or zone_id
        self.current_count: int = 0
        self.total_entries: int = 0
        self.total_exits: int = 0
        self.track_ids_inside: set[int] = set()
        self.class_counts: dict[str, int] = defaultdict(int)

    def contains(self, point: tuple[float, float]) -> bool:
        return MathUtils.point_in_polygon(point, [(float(p[0]), float(p[1])) for p in self.points])

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "current_count": self.current_count,
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "track_ids_inside": list(self.track_ids_inside),
            "class_counts": dict(self.class_counts),
        }


class ZoneAnalytics:
    def __init__(self, zones: list[dict[str, Any]] | None = None) -> None:
        self._zones: dict[str, Zone] = {}
        if zones:
            for z in zones:
                self.add_zone(
                    zone_id=z.get("id", f"zone_{len(self._zones)}"),
                    points=z["points"],
                    name=z.get("name", ""),
                )

    def add_zone(self, zone_id: str, points: list[tuple[int, int]], name: str = "") -> Zone:
        zone = Zone(zone_id=zone_id, points=points, name=name)
        self._zones[zone_id] = zone
        return zone

    def remove_zone(self, zone_id: str) -> None:
        self._zones.pop(zone_id, None)

    @property
    def zones(self) -> dict[str, Zone]:
        return self._zones

    def update(self, tracks: list[Track]) -> list[dict[str, Any]]:
        events = []

        for zone in self._zones.values():
            current_inside = set()

            for track in tracks:
                center = track.center
                if zone.contains(center):
                    current_inside.add(track.track_id)

                    if track.track_id not in zone.track_ids_inside:
                        zone.total_entries += 1
                        zone.class_counts[track.class_name] += 1
                        events.append(
                            {
                                "type": "zone_enter",
                                "zone_id": zone.zone_id,
                                "zone_name": zone.name,
                                "track_id": track.track_id,
                                "class_name": track.class_name,
                            }
                        )

            exited = zone.track_ids_inside - current_inside
            for tid in exited:
                zone.total_exits += 1
                events.append(
                    {
                        "type": "zone_exit",
                        "zone_id": zone.zone_id,
                        "zone_name": zone.name,
                        "track_id": tid,
                    }
                )

            zone.track_ids_inside = current_inside
            zone.current_count = len(current_inside)

        return events

    def get_zone_stats(self, zone_id: str) -> dict[str, Any] | None:
        zone = self._zones.get(zone_id)
        return zone.to_dict() if zone else None

    def get_all_stats(self) -> dict[str, Any]:
        return {zid: z.to_dict() for zid, z in self._zones.items()}

    def get_occupancy(self) -> dict[str, int]:
        return {zid: z.current_count for zid, z in self._zones.items()}

    def reset(self) -> None:
        for zone in self._zones.values():
            zone.current_count = 0
            zone.total_entries = 0
            zone.total_exits = 0
            zone.track_ids_inside.clear()
            zone.class_counts.clear()
