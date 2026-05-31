from __future__ import annotations

import numpy as np

from src.core.base import ComponentState, DetectionResult
from src.core.registry import tracker_registry
from src.tracking.association import fuse_iou_score, iou_distance, linear_assignment
from src.tracking.base_tracker import BaseObjectTracker
from src.tracking.kalman_filter import KalmanFilterXYAH
from src.tracking.track import Track, TrackState
from src.utils.logger import get_logger

logger = get_logger(__name__)


@tracker_registry.register("botsort")
class BoTSORTTracker(BaseObjectTracker):
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        with_reid: bool = False,
        fuse_score: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            name=name or "BoTSORT",
        )
        self._track_high_thresh = track_high_thresh
        self._track_low_thresh = track_low_thresh
        self._new_track_thresh = new_track_thresh
        self._match_thresh = match_thresh
        self._proximity_thresh = proximity_thresh
        self._appearance_thresh = appearance_thresh
        self._with_reid = with_reid
        self._fuse_score = fuse_score
        self._kf = KalmanFilterXYAH()
        self._kalman_states: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def initialize(self) -> None:
        self.set_state(ComponentState.READY)
        logger.info(
            "botsort_initialized",
            with_reid=self._with_reid,
            fuse_score=self._fuse_score,
        )

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

    def _predict_tracks(self) -> None:
        for track in self._tracks:
            if track.track_id in self._kalman_states:
                mean, cov = self._kalman_states[track.track_id]
                mean, cov = self._kf.predict(mean, cov)
                self._kalman_states[track.track_id] = (mean, cov)
                track.box = self._xyah_to_box(mean)

    def update(self, detections: DetectionResult, frame: np.ndarray | None = None) -> list[Track]:
        self._frame_count += 1
        self._predict_tracks()

        if detections.num_detections == 0:
            for track in self._tracks:
                track.mark_missed()
                if track.time_since_update > self._max_age:
                    track.state = TrackState.DELETED
            self._remove_deleted_tracks()
            return self.confirmed_tracks

        high_mask = detections.scores >= self._track_high_thresh
        low_mask = (detections.scores >= self._track_low_thresh) & (~high_mask)

        high_boxes = detections.boxes[high_mask]
        high_scores = detections.scores[high_mask]
        high_class_ids = detections.class_ids[high_mask]
        high_names = [n for n, m in zip(detections.class_names or [], high_mask) if m]

        low_boxes = detections.boxes[low_mask]
        low_scores = detections.scores[low_mask]
        low_class_ids = detections.class_ids[low_mask]
        low_names = [n for n, m in zip(detections.class_names or [], low_mask) if m]

        confirmed = [t for t in self._tracks if t.is_confirmed]
        unconfirmed = [t for t in self._tracks if not t.is_confirmed]

        confirmed_boxes = np.array([t.box for t in confirmed]) if confirmed else np.empty((0, 4))

        if len(confirmed_boxes) > 0 and len(high_boxes) > 0:
            cost = iou_distance(confirmed_boxes, high_boxes)
            if self._fuse_score:
                cost = fuse_iou_score(cost, high_scores)
            matches_1, unmatched_t1, unmatched_d1 = linear_assignment(cost, threshold=1.0 - self._match_thresh)
        else:
            matches_1 = []
            unmatched_t1 = list(range(len(confirmed)))
            unmatched_d1 = list(range(len(high_boxes)))

        for t_idx, d_idx in matches_1:
            self._apply_update(
                confirmed[t_idx],
                high_boxes[d_idx],
                float(high_scores[d_idx]),
                int(high_class_ids[d_idx]),
                high_names[d_idx] if high_names else "",
            )

        remaining = [confirmed[i] for i in unmatched_t1]
        remaining_boxes = np.array([t.box for t in remaining]) if remaining else np.empty((0, 4))

        if len(remaining_boxes) > 0 and len(low_boxes) > 0:
            cost_2 = iou_distance(remaining_boxes, low_boxes)
            matches_2, unmatched_t2, _ = linear_assignment(cost_2, threshold=1.0 - 0.5)
        else:
            matches_2 = []
            unmatched_t2 = list(range(len(remaining)))

        for t_idx, d_idx in matches_2:
            self._apply_update(
                remaining[t_idx],
                low_boxes[d_idx],
                float(low_scores[d_idx]),
                int(low_class_ids[d_idx]),
                low_names[d_idx] if low_names else "",
            )

        for t_idx in unmatched_t2:
            track = remaining[t_idx]
            track.mark_missed()
            if track.time_since_update > self._max_age:
                track.state = TrackState.DELETED

        unconf_boxes = np.array([t.box for t in unconfirmed]) if unconfirmed else np.empty((0, 4))
        rem_high = high_boxes[unmatched_d1] if unmatched_d1 else np.empty((0, 4))

        if len(unconf_boxes) > 0 and len(rem_high) > 0:
            cost_3 = iou_distance(unconf_boxes, rem_high)
            matches_3, unmatched_unc, unmatched_new = linear_assignment(cost_3, threshold=1.0 - 0.7)
        else:
            matches_3 = []
            unmatched_unc = list(range(len(unconfirmed)))
            unmatched_new = list(range(len(rem_high)))

        for t_idx, d_idx in matches_3:
            actual_idx = unmatched_d1[d_idx]
            self._apply_update(
                unconfirmed[t_idx],
                high_boxes[actual_idx],
                float(high_scores[actual_idx]),
                int(high_class_ids[actual_idx]),
                high_names[actual_idx] if high_names else "",
            )

        for t_idx in unmatched_unc:
            unconfirmed[t_idx].state = TrackState.DELETED

        for d_idx in unmatched_new:
            actual_idx = unmatched_d1[d_idx]
            score = float(high_scores[actual_idx])
            if score >= self._new_track_thresh:
                track = self._create_track(
                    high_boxes[actual_idx],
                    score,
                    int(high_class_ids[actual_idx]),
                    high_names[actual_idx] if high_names else "",
                )
                self._tracks.append(track)
                self._total_tracks += 1
                measurement = self._box_to_xyah(high_boxes[actual_idx])
                mean, cov = self._kf.initiate(measurement)
                self._kalman_states[track.track_id] = (mean, cov)

        self._remove_deleted_tracks()
        return self.confirmed_tracks

    def _apply_update(self, track: Track, box: np.ndarray, score: float, class_id: int, class_name: str) -> None:
        track.update_box(box, score, class_id, class_name)
        measurement = self._box_to_xyah(box)
        if track.track_id in self._kalman_states:
            mean, cov = self._kalman_states[track.track_id]
            mean, cov = self._kf.update(mean, cov, measurement)
            self._kalman_states[track.track_id] = (mean, cov)
        else:
            mean, cov = self._kf.initiate(measurement)
            self._kalman_states[track.track_id] = (mean, cov)
        if track.hits >= self._min_hits:
            track.state = TrackState.CONFIRMED

    def shutdown(self) -> None:
        self.reset()
        self._kalman_states.clear()
        self.set_state(ComponentState.STOPPED)
