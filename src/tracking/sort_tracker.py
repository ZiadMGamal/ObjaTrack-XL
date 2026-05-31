from __future__ import annotations

from typing import Any

import numpy as np

from src.core.base import ComponentState, DetectionResult
from src.core.registry import tracker_registry
from src.tracking.base_tracker import BaseObjectTracker
from src.tracking.track import Track, TrackState
from src.tracking.kalman_filter import KalmanFilterXYAH
from src.tracking.association import iou_distance, linear_assignment
from src.utils.logger import get_logger

logger = get_logger(__name__)


@tracker_registry.register("sort")
class SORTTracker(BaseObjectTracker):

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        name: str | None = None,
    ) -> None:
        super().__init__(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            name=name or "SORT",
        )
        self._kalman_filters: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._kf = KalmanFilterXYAH()

    def initialize(self) -> None:
        self.set_state(ComponentState.READY)
        logger.info("sort_initialized", max_age=self._max_age, min_hits=self._min_hits)

    def _box_to_xyah(self, box: np.ndarray) -> np.ndarray:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]
        return np.array([cx, cy, w / h if h > 0 else 0, h])

    def _xyah_to_box(self, xyah: np.ndarray) -> np.ndarray:
        cx, cy, a, h = xyah[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def update(self, detections: DetectionResult, frame: np.ndarray | None = None) -> list[Track]:
        self._frame_count += 1

        for track in self._tracks:
            if track.track_id in self._kalman_filters:
                mean, cov = self._kalman_filters[track.track_id]
                mean, cov = self._kf.predict(mean, cov)
                self._kalman_filters[track.track_id] = (mean, cov)
                track.box = self._xyah_to_box(mean)

        if detections.num_detections == 0:
            for track in self._tracks:
                track.mark_missed()
                if track.time_since_update > self._max_age:
                    track.state = TrackState.DELETED
            self._remove_deleted_tracks()
            return self.confirmed_tracks

        track_boxes = np.array([t.box for t in self._tracks]) if self._tracks else np.empty((0, 4))
        det_boxes = detections.boxes

        if len(track_boxes) > 0 and len(det_boxes) > 0:
            cost_matrix = iou_distance(track_boxes, det_boxes)
            matches, unmatched_tracks, unmatched_dets = linear_assignment(
                cost_matrix, threshold=1.0 - self._iou_threshold
            )
        else:
            matches = []
            unmatched_tracks = list(range(len(self._tracks)))
            unmatched_dets = list(range(detections.num_detections))

        for track_idx, det_idx in matches:
            track = self._tracks[track_idx]
            box = detections.boxes[det_idx]
            score = float(detections.scores[det_idx])
            class_id = int(detections.class_ids[det_idx])
            class_name = detections.class_names[det_idx] if detections.class_names else ""

            track.update_box(box, score, class_id, class_name)

            measurement = self._box_to_xyah(box)
            if track.track_id in self._kalman_filters:
                mean, cov = self._kalman_filters[track.track_id]
                mean, cov = self._kf.update(mean, cov, measurement)
                self._kalman_filters[track.track_id] = (mean, cov)

            if track.hits >= self._min_hits:
                track.state = TrackState.CONFIRMED

        for track_idx in unmatched_tracks:
            track = self._tracks[track_idx]
            track.mark_missed()
            if track.time_since_update > self._max_age:
                track.state = TrackState.DELETED
                if track.track_id in self._kalman_filters:
                    del self._kalman_filters[track.track_id]

        for det_idx in unmatched_dets:
            box = detections.boxes[det_idx]
            score = float(detections.scores[det_idx])
            class_id = int(detections.class_ids[det_idx])
            class_name = detections.class_names[det_idx] if detections.class_names else ""

            track = self._create_track(box, score, class_id, class_name)
            self._tracks.append(track)
            self._total_tracks += 1

            measurement = self._box_to_xyah(box)
            mean, cov = self._kf.initiate(measurement)
            self._kalman_filters[track.track_id] = (mean, cov)

        self._remove_deleted_tracks()
        return self.confirmed_tracks

    def shutdown(self) -> None:
        self.reset()
        self._kalman_filters.clear()
        self.set_state(ComponentState.STOPPED)
