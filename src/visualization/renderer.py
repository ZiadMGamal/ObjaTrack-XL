from __future__ import annotations

import cv2
import numpy as np

from src.core.base import DetectionResult
from src.tracking.track import Track
from src.utils.logger import get_logger
from src.visualization.color_palette import ColorPalette

logger = get_logger(__name__)


class OverlayRenderer:
    def __init__(
        self,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_confidence: bool = True,
        show_tracks: bool = True,
        show_trajectory: bool = True,
        trajectory_length: int = 30,
        box_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 1,
        label_padding: int = 5,
    ) -> None:
        self._show_boxes = show_boxes
        self._show_labels = show_labels
        self._show_confidence = show_confidence
        self._show_tracks = show_tracks
        self._show_trajectory = show_trajectory
        self._trajectory_length = trajectory_length
        self._box_thickness = box_thickness
        self._font_scale = font_scale
        self._font_thickness = font_thickness
        self._label_padding = label_padding
        self._palette = ColorPalette()
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    @property
    def palette(self) -> ColorPalette:
        return self._palette

    def draw_detections(self, frame: np.ndarray, detections: DetectionResult) -> np.ndarray:
        output = frame.copy()

        for i in range(detections.num_detections):
            box = detections.boxes[i].astype(int)
            score = detections.scores[i]
            class_id = int(detections.class_ids[i])
            class_name = detections.class_names[i] if detections.class_names else f"cls_{class_id}"

            color = self._palette.get_class_color(class_name)

            if self._show_boxes:
                cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]), color, self._box_thickness)

            if self._show_labels:
                label = class_name
                if self._show_confidence:
                    label = f"{class_name} {score:.2f}"
                output = self._draw_label(output, label, (box[0], box[1]), color)

        return output

    def draw_tracks(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        output = frame.copy()

        for track in tracks:
            color = self._palette.get_color(track.track_id)
            box = track.box.astype(int)

            if self._show_boxes:
                cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]), color, self._box_thickness)

                corner_len = min(20, (box[2] - box[0]) // 4, (box[3] - box[1]) // 4)
                cv2.line(output, (box[0], box[1]), (box[0] + corner_len, box[1]), color, self._box_thickness + 1)
                cv2.line(output, (box[0], box[1]), (box[0], box[1] + corner_len), color, self._box_thickness + 1)
                cv2.line(output, (box[2], box[1]), (box[2] - corner_len, box[1]), color, self._box_thickness + 1)
                cv2.line(output, (box[2], box[1]), (box[2], box[1] + corner_len), color, self._box_thickness + 1)
                cv2.line(output, (box[0], box[3]), (box[0] + corner_len, box[3]), color, self._box_thickness + 1)
                cv2.line(output, (box[0], box[3]), (box[0], box[3] - corner_len), color, self._box_thickness + 1)
                cv2.line(output, (box[2], box[3]), (box[2] - corner_len, box[3]), color, self._box_thickness + 1)
                cv2.line(output, (box[2], box[3]), (box[2], box[3] - corner_len), color, self._box_thickness + 1)

            if self._show_labels:
                parts = [f"ID:{track.track_id}"]
                if track.class_name:
                    parts.append(track.class_name)
                if self._show_confidence:
                    parts.append(f"{track.score:.2f}")
                label = " | ".join(parts)
                output = self._draw_label(output, label, (box[0], box[1]), color)

            if self._show_trajectory and track.trajectory:
                output = self._draw_trajectory(output, track, color)

        return output

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ) -> np.ndarray:
        (text_w, text_h), baseline = cv2.getTextSize(label, self._font, self._font_scale, self._font_thickness)

        x, y = position
        pad = self._label_padding

        label_y1 = max(0, y - text_h - 2 * pad)
        label_y2 = y
        label_x1 = x
        label_x2 = x + text_w + 2 * pad

        overlay = frame.copy()
        cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        text_color = self._palette.get_contrast_color(color)
        cv2.putText(
            frame,
            label,
            (label_x1 + pad, label_y2 - pad),
            self._font,
            self._font_scale,
            text_color,
            self._font_thickness,
            cv2.LINE_AA,
        )

        return frame

    def _draw_trajectory(
        self,
        frame: np.ndarray,
        track: Track,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        trajectory = track.trajectory[-self._trajectory_length :]
        if len(trajectory) < 2:
            return frame

        points = np.array(trajectory, dtype=np.int32)

        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(alpha * 3))
            fade_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, tuple(points[i - 1]), tuple(points[i]), fade_color, thickness, cv2.LINE_AA)

        if len(points) > 0:
            cv2.circle(frame, tuple(points[-1]), 4, color, -1, cv2.LINE_AA)

        return frame

    def draw_counting_line(
        self,
        frame: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
        count_in: int = 0,
        count_out: int = 0,
    ) -> np.ndarray:
        cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)

        mid_x = (start[0] + end[0]) // 2
        mid_y = (start[1] + end[1]) // 2

        label = f"IN:{count_in} OUT:{count_out}"
        cv2.putText(frame, label, (mid_x - 40, mid_y - 10), self._font, 0.7, color, 2, cv2.LINE_AA)

        return frame

    def draw_zone(
        self,
        frame: np.ndarray,
        points: list[tuple[int, int]],
        color: tuple[int, int, int] = (0, 200, 200),
        alpha: float = 0.3,
        label: str | None = None,
    ) -> np.ndarray:
        pts = np.array(points, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)

        if label:
            cx = int(np.mean([p[0] for p in points]))
            cy = int(np.mean([p[1] for p in points]))
            cv2.putText(frame, label, (cx - 20, cy), self._font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return frame
