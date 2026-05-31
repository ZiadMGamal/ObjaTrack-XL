from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from src.core.base import BaseTracker, DetectionResult
from src.tracking.track import Track, TrackState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseObjectTracker(BaseTracker):

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        name: str | None = None,
    ) -> None:
        super().__init__(max_age=max_age, min_hits=min_hits, name=name)
        self._iou_threshold = iou_threshold
        self._tracks: list[Track] = []
        self._next_id: int = 1
        self._deleted_tracks: list[Track] = []
        self._max_deleted_history: int = 1000

    @property
    def iou_threshold(self) -> float:
        return self._iou_threshold

    @property
    def tracks(self) -> list[Track]:
        return self._tracks

    @property
    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.is_confirmed]

    @property
    def tentative_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.is_tentative]

    @property
    def deleted_tracks(self) -> list[Track]:
        return self._deleted_tracks

    def _get_next_id(self) -> int:
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def _create_track(
        self,
        box: np.ndarray,
        score: float,
        class_id: int,
        class_name: str = "",
    ) -> Track:
        track = Track(
            track_id=self._get_next_id(),
            box=box.copy(),
            score=score,
            class_id=class_id,
            class_name=class_name,
            state=TrackState.TENTATIVE,
            age=1,
            hits=1,
            time_since_update=0,
        )
        track.trajectory.append(track.center)
        return track

    def _remove_deleted_tracks(self) -> None:
        surviving = []
        for track in self._tracks:
            if track.is_deleted:
                self._deleted_tracks.append(track)
            else:
                surviving.append(track)

        self._tracks = surviving
        self._active_tracks = len(self._tracks)

        if len(self._deleted_tracks) > self._max_deleted_history:
            self._deleted_tracks = self._deleted_tracks[-self._max_deleted_history:]

    @abstractmethod
    def update(self, detections: DetectionResult, frame: np.ndarray | None = None) -> list[Track]:
        ...

    def reset(self) -> None:
        self._tracks.clear()
        self._deleted_tracks.clear()
        self._next_id = 1
        self._frame_count = 0
        self._active_tracks = 0
        self._total_tracks = 0
        logger.info("tracker_reset", name=self._name)

    def get_active_tracks_as_array(self) -> np.ndarray:
        confirmed = self.confirmed_tracks
        if not confirmed:
            return np.empty((0, 7))

        result = np.zeros((len(confirmed), 7))
        for i, track in enumerate(confirmed):
            result[i, :4] = track.box
            result[i, 4] = track.track_id
            result[i, 5] = track.class_id
            result[i, 6] = track.score

        return result
