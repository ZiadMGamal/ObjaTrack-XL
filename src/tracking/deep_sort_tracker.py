from __future__ import annotations

from typing import Any

import numpy as np

from src.core.base import ComponentState, DetectionResult
from src.core.registry import tracker_registry
from src.tracking.base_tracker import BaseObjectTracker
from src.tracking.track import Track, TrackState
from src.tracking.kalman_filter import KalmanFilterXYAH
from src.tracking.association import iou_distance, cosine_distance, linear_assignment
from src.utils.logger import get_logger

logger = get_logger(__name__)


@tracker_registry.register("deepsort")
class DeepSORTTracker(BaseObjectTracker):

    def __init__(
        self,
        max_age: int = 70,
        min_hits: int = 3,
        iou_threshold: float = 0.7,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 100,
        name: str | None = None,
    ) -> None:
        super().__init__(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            name=name or "DeepSORT",
        )
        self._max_cosine_distance = max_cosine_distance
        self._nn_budget = nn_budget
        self._kf = KalmanFilterXYAH()
        self._kalman_states: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def initialize(self) -> None:
        self.set_state(ComponentState.READY)
        logger.info(
            "deepsort_initialized",
            max_cosine_distance=self._max_cosine_distance,
            nn_budget=self._nn_budget,
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

    def _extract_features(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        features = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                features.append(np.zeros(128))
                continue

            crop = frame[y1:y2, x1:x2]
            feature = np.mean(crop.reshape(-1, 3).astype(np.float32), axis=0)
            feature = np.tile(feature, 128 // 3 + 1)[:128]
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            features.append(feature)

        return np.array(features) if features else np.empty((0, 128))

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

        det_features = None
        if frame is not None:
            det_features = self._extract_features(frame, detections.boxes)

        confirmed = [t for t in self._tracks if t.is_confirmed]
        unconfirmed = [t for t in self._tracks if not t.is_confirmed]

        confirmed_boxes = np.array([t.box for t in confirmed]) if confirmed else np.empty((0, 4))

        if len(confirmed_boxes) > 0 and detections.num_detections > 0:
            iou_cost = iou_distance(confirmed_boxes, detections.boxes)

            if det_features is not None and len(det_features) > 0:
                track_features = [t.get_latest_feature() for t in confirmed]
                valid_features = []
                for f in track_features:
                    if f is not None and len(f) > 0:
                        valid_features.append(f)
                    else:
                        valid_features.append(np.zeros(128))

                appearance_cost = cosine_distance(valid_features, det_features)
                cost = 0.5 * iou_cost + 0.5 * appearance_cost
            else:
                cost = iou_cost

            matches, unmatched_tracks, unmatched_dets = linear_assignment(
                cost, threshold=1.0 - self._iou_threshold
            )
        else:
            matches = []
            unmatched_tracks = list(range(len(confirmed)))
            unmatched_dets = list(range(detections.num_detections))

        for t_idx, d_idx in matches:
            track = confirmed[t_idx]
            box = detections.boxes[d_idx]
            score = float(detections.scores[d_idx])
            class_id = int(detections.class_ids[d_idx])
            name = detections.class_names[d_idx] if detections.class_names else ""

            track.update_box(box, score, class_id, name)

            if det_features is not None and d_idx < len(det_features):
                track.add_feature(det_features[d_idx], self._nn_budget)

            measurement = self._box_to_xyah(box)
            if track.track_id in self._kalman_states:
                mean, cov = self._kalman_states[track.track_id]
                mean, cov = self._kf.update(mean, cov, measurement)
                self._kalman_states[track.track_id] = (mean, cov)

        for t_idx in unmatched_tracks:
            track = confirmed[t_idx]
            track.mark_missed()
            if track.time_since_update > self._max_age:
                track.state = TrackState.DELETED

        unconf_boxes = np.array([t.box for t in unconfirmed]) if unconfirmed else np.empty((0, 4))
        rem_det_boxes = detections.boxes[unmatched_dets] if unmatched_dets else np.empty((0, 4))

        if len(unconf_boxes) > 0 and len(rem_det_boxes) > 0:
            cost_uc = iou_distance(unconf_boxes, rem_det_boxes)
            m_uc, um_uc, um_d = linear_assignment(cost_uc, threshold=0.7)
        else:
            m_uc = []
            um_uc = list(range(len(unconfirmed)))
            um_d = list(range(len(rem_det_boxes)))

        for t_idx, d_idx in m_uc:
            actual_d = unmatched_dets[d_idx]
            track = unconfirmed[t_idx]
            box = detections.boxes[actual_d]
            score = float(detections.scores[actual_d])
            class_id = int(detections.class_ids[actual_d])
            name = detections.class_names[actual_d] if detections.class_names else ""
            track.update_box(box, score, class_id, name)

            if det_features is not None and actual_d < len(det_features):
                track.add_feature(det_features[actual_d], self._nn_budget)

            measurement = self._box_to_xyah(box)
            if track.track_id in self._kalman_states:
                mean, cov = self._kalman_states[track.track_id]
                mean, cov = self._kf.update(mean, cov, measurement)
                self._kalman_states[track.track_id] = (mean, cov)

            if track.hits >= self._min_hits:
                track.state = TrackState.CONFIRMED

        for t_idx in um_uc:
            unconfirmed[t_idx].state = TrackState.DELETED

        for d_idx in um_d:
            actual_d = unmatched_dets[d_idx]
            box = detections.boxes[actual_d]
            score = float(detections.scores[actual_d])
            class_id = int(detections.class_ids[actual_d])
            name = detections.class_names[actual_d] if detections.class_names else ""

            track = self._create_track(box, score, class_id, name)
            self._tracks.append(track)
            self._total_tracks += 1

            if det_features is not None and actual_d < len(det_features):
                track.add_feature(det_features[actual_d], self._nn_budget)

            measurement = self._box_to_xyah(box)
            mean, cov = self._kf.initiate(measurement)
            self._kalman_states[track.track_id] = (mean, cov)

        self._remove_deleted_tracks()
        return self.confirmed_tracks

    def shutdown(self) -> None:
        self.reset()
        self._kalman_states.clear()
        self.set_state(ComponentState.STOPPED)
