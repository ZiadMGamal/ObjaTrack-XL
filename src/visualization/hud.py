from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HeadsUpDisplay:

    def __init__(
        self,
        position: str = "top-left",
        font_scale: float = 0.5,
        font_thickness: int = 1,
        line_spacing: int = 22,
        padding: int = 10,
        bg_alpha: float = 0.6,
        bg_color: tuple[int, int, int] = (20, 20, 20),
        text_color: tuple[int, int, int] = (220, 220, 220),
        accent_color: tuple[int, int, int] = (0, 200, 255),
    ) -> None:
        self._position = position
        self._font_scale = font_scale
        self._font_thickness = font_thickness
        self._line_spacing = line_spacing
        self._padding = padding
        self._bg_alpha = bg_alpha
        self._bg_color = bg_color
        self._text_color = text_color
        self._accent_color = accent_color
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._start_time = time.time()

    def render(
        self,
        frame: np.ndarray,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        num_detections: int = 0,
        num_tracks: int = 0,
        frame_id: int = 0,
        model_name: str = "",
        tracker_name: str = "",
        resolution: str = "",
        extra_info: dict[str, Any] | None = None,
    ) -> np.ndarray:
        lines: list[tuple[str, str]] = []

        elapsed = time.time() - self._start_time
        mins, secs = divmod(int(elapsed), 60)
        hours, mins = divmod(mins, 60)

        lines.append(("ObjaTrack-XL", ""))
        lines.append(("Time", f"{hours:02d}:{mins:02d}:{secs:02d}"))
        lines.append(("Frame", str(frame_id)))

        if resolution:
            lines.append(("Resolution", resolution))

        if model_name:
            lines.append(("Model", model_name))

        if tracker_name:
            lines.append(("Tracker", tracker_name))

        lines.append(("FPS", f"{fps:.1f}"))
        lines.append(("Latency", f"{latency_ms:.1f}ms"))
        lines.append(("Detections", str(num_detections)))
        lines.append(("Tracks", str(num_tracks)))

        if extra_info:
            for key, value in extra_info.items():
                lines.append((key, str(value)))

        return self._draw_panel(frame, lines)

    def _draw_panel(self, frame: np.ndarray, lines: list[tuple[str, str]]) -> np.ndarray:
        max_width = 0
        for label, value in lines:
            text = f"{label}: {value}" if value else label
            (tw, _), _ = cv2.getTextSize(text, self._font, self._font_scale, self._font_thickness)
            max_width = max(max_width, tw)

        panel_w = max_width + self._padding * 3
        panel_h = len(lines) * self._line_spacing + self._padding * 2

        h, w = frame.shape[:2]

        if self._position == "top-left":
            x, y = self._padding, self._padding
        elif self._position == "top-right":
            x, y = w - panel_w - self._padding, self._padding
        elif self._position == "bottom-left":
            x, y = self._padding, h - panel_h - self._padding
        else:
            x, y = w - panel_w - self._padding, h - panel_h - self._padding

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), self._bg_color, -1)
        cv2.addWeighted(overlay, self._bg_alpha, frame, 1 - self._bg_alpha, 0, frame)

        cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), self._accent_color, 1)
        cv2.line(frame, (x, y + self._line_spacing + 5), (x + panel_w, y + self._line_spacing + 5), self._accent_color, 1)

        for i, (label, value) in enumerate(lines):
            text_y = y + self._padding + (i + 1) * self._line_spacing - 4

            if i == 0:
                cv2.putText(frame, label, (x + self._padding, text_y),
                            self._font, self._font_scale + 0.1, self._accent_color,
                            self._font_thickness + 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"{label}:", (x + self._padding, text_y),
                            self._font, self._font_scale, self._text_color,
                            self._font_thickness, cv2.LINE_AA)

                (lw, _), _ = cv2.getTextSize(f"{label}: ", self._font, self._font_scale, self._font_thickness)
                value_color = self._accent_color if label in ("FPS", "Latency") else (180, 255, 180)
                cv2.putText(frame, value, (x + self._padding + lw, text_y),
                            self._font, self._font_scale, value_color,
                            self._font_thickness, cv2.LINE_AA)

        return frame

    def render_mini_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        label = f"FPS: {fps:.1f}"
        h, w = frame.shape[:2]
        x, y = w - 120, 30

        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - 20), (x + 110, y + 5), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        color = (0, 255, 0) if fps >= 25 else (0, 200, 255) if fps >= 15 else (0, 0, 255)
        cv2.putText(frame, label, (x, y), self._font, 0.6, color, 2, cv2.LINE_AA)

        return frame
