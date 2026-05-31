from __future__ import annotations

import time
from collections import defaultdict
from enum import Enum
from typing import Any

from src.tracking.track import Track
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    ENTER = "enter"
    EXIT = "exit"
    LOITER = "loiter"
    SPEED_VIOLATION = "speed_violation"
    WRONG_WAY = "wrong_way"
    CROWD_FORMATION = "crowd_formation"
    OBJECT_LEFT = "object_left"
    OBJECT_REMOVED = "object_removed"


class EventDetector:
    def __init__(
        self,
        loiter_threshold: float = 30.0,
        crowd_threshold: int = 5,
        crowd_radius: float = 100.0,
        speed_limit: float = 60.0,
        stationary_threshold: float = 5.0,
        stationary_duration: float = 60.0,
    ) -> None:
        self._loiter_threshold = loiter_threshold
        self._crowd_threshold = crowd_threshold
        self._crowd_radius = crowd_radius
        self._speed_limit = speed_limit
        self._stationary_threshold = stationary_threshold
        self._stationary_duration = stationary_duration
        self._track_entry_times: dict[int, float] = {}
        self._track_positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self._stationary_start: dict[int, float] = {}
        self._alerted_loiter: set[int] = set()
        self._alerted_stationary: set[int] = set()
        self._event_history: list[dict[str, Any]] = []
        self._max_history: int = 10000

    @property
    def event_history(self) -> list[dict[str, Any]]:
        return self._event_history

    def update(
        self,
        tracks: list[Track],
        speeds: dict[int, float] | None = None,
    ) -> list[dict[str, Any]]:
        events = []
        current_time = time.time()
        active_ids = set()

        for track in tracks:
            active_ids.add(track.track_id)

            if track.track_id not in self._track_entry_times:
                self._track_entry_times[track.track_id] = current_time
                events.append(self._create_event(EventType.ENTER, track))

            self._track_positions[track.track_id].append(track.center)
            if len(self._track_positions[track.track_id]) > 100:
                self._track_positions[track.track_id] = self._track_positions[track.track_id][-100:]

        departed = set(self._track_entry_times.keys()) - active_ids
        for tid in departed:
            events.append(
                {
                    "type": EventType.EXIT.value,
                    "track_id": tid,
                    "timestamp": current_time,
                    "duration": current_time - self._track_entry_times.get(tid, current_time),
                }
            )
            self._cleanup_track(tid)

        loiter_events = self._detect_loitering(tracks, current_time)
        events.extend(loiter_events)

        if speeds:
            speed_events = self._detect_speed_violations(tracks, speeds)
            events.extend(speed_events)

        crowd_events = self._detect_crowds(tracks)
        events.extend(crowd_events)

        stationary_events = self._detect_stationary_objects(tracks, current_time)
        events.extend(stationary_events)

        self._event_history.extend(events)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history :]

        return events

    def _detect_loitering(self, tracks: list[Track], current_time: float) -> list[dict[str, Any]]:
        events = []
        for track in tracks:
            if track.track_id in self._alerted_loiter:
                continue

            entry_time = self._track_entry_times.get(track.track_id, current_time)
            duration = current_time - entry_time

            if duration >= self._loiter_threshold:
                positions = self._track_positions.get(track.track_id, [])
                if len(positions) >= 2:
                    xs = [p[0] for p in positions]
                    ys = [p[1] for p in positions]
                    spread = max(max(xs) - min(xs), max(ys) - min(ys))
                    if spread < 200:
                        self._alerted_loiter.add(track.track_id)
                        event = self._create_event(EventType.LOITER, track)
                        event["duration"] = round(duration, 1)
                        event["spread"] = round(spread, 1)
                        events.append(event)

        return events

    def _detect_speed_violations(self, tracks: list[Track], speeds: dict[int, float]) -> list[dict[str, Any]]:
        events = []
        for track in tracks:
            speed = speeds.get(track.track_id, 0.0)
            if speed > self._speed_limit:
                event = self._create_event(EventType.SPEED_VIOLATION, track)
                event["speed"] = round(speed, 1)
                event["limit"] = self._speed_limit
                events.append(event)
        return events

    def _detect_crowds(self, tracks: list[Track]) -> list[dict[str, Any]]:
        events = []
        if len(tracks) < self._crowd_threshold:
            return events

        centers = [(t.center, t.track_id) for t in tracks]
        for i, (c1, tid1) in enumerate(centers):
            nearby = 0
            for j, (c2, tid2) in enumerate(centers):
                if i == j:
                    continue
                dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
                if dist < self._crowd_radius:
                    nearby += 1

            if nearby >= self._crowd_threshold - 1:
                events.append(
                    {
                        "type": EventType.CROWD_FORMATION.value,
                        "center": c1,
                        "count": nearby + 1,
                        "timestamp": time.time(),
                    }
                )
                break

        return events

    def _detect_stationary_objects(self, tracks: list[Track], current_time: float) -> list[dict[str, Any]]:
        events = []
        for track in tracks:
            if track.track_id in self._alerted_stationary:
                continue

            if track.speed < self._stationary_threshold:
                if track.track_id not in self._stationary_start:
                    self._stationary_start[track.track_id] = current_time
                elif current_time - self._stationary_start[track.track_id] >= self._stationary_duration:
                    self._alerted_stationary.add(track.track_id)
                    event = self._create_event(EventType.OBJECT_LEFT, track)
                    event["stationary_duration"] = round(current_time - self._stationary_start[track.track_id], 1)
                    events.append(event)
            else:
                self._stationary_start.pop(track.track_id, None)

        return events

    def _create_event(self, event_type: EventType, track: Track) -> dict[str, Any]:
        return {
            "type": event_type.value,
            "track_id": track.track_id,
            "class_name": track.class_name,
            "position": track.center,
            "timestamp": time.time(),
        }

    def _cleanup_track(self, track_id: int) -> None:
        self._track_entry_times.pop(track_id, None)
        self._track_positions.pop(track_id, None)
        self._stationary_start.pop(track_id, None)
        self._alerted_loiter.discard(track_id)
        self._alerted_stationary.discard(track_id)

    def get_statistics(self) -> dict[str, Any]:
        type_counts = defaultdict(int)
        for event in self._event_history:
            type_counts[event["type"]] += 1

        return {
            "total_events": len(self._event_history),
            "event_counts": dict(type_counts),
            "active_tracks": len(self._track_entry_times),
        }

    def reset(self) -> None:
        self._track_entry_times.clear()
        self._track_positions.clear()
        self._stationary_start.clear()
        self._alerted_loiter.clear()
        self._alerted_stationary.clear()
        self._event_history.clear()
