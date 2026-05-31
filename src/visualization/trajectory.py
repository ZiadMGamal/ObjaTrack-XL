from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from src.tracking.track import Track
from src.visualization.color_palette import ColorPalette


class TrajectoryVisualizer:
    def __init__(
        self,
        max_length: int = 100,
        fade: bool = True,
        line_thickness: int = 2,
        dot_radius: int = 3,
        show_direction: bool = True,
        smooth: bool = True,
        smooth_window: int = 5,
    ) -> None:
        self._max_length = max_length
        self._fade = fade
        self._line_thickness = line_thickness
        self._dot_radius = dot_radius
        self._show_direction = show_direction
        self._smooth = smooth
        self._smooth_window = smooth_window
        self._palette = ColorPalette()
        self._trajectories: dict[int, list[tuple[float, float]]] = defaultdict(list)

    def update(self, tracks: list[Track]) -> None:
        active_ids = set()
        for track in tracks:
            active_ids.add(track.track_id)
            self._trajectories[track.track_id].append(track.center)
            if len(self._trajectories[track.track_id]) > self._max_length:
                self._trajectories[track.track_id] = self._trajectories[track.track_id][-self._max_length :]

        stale_ids = [tid for tid in self._trajectories if tid not in active_ids]
        for tid in stale_ids:
            if len(self._trajectories[tid]) > 0:
                self._trajectories[tid] = self._trajectories[tid][:-1]
                if not self._trajectories[tid]:
                    del self._trajectories[tid]

    def draw(self, frame: np.ndarray, tracks: list[Track] | None = None) -> np.ndarray:
        if tracks is not None:
            self.update(tracks)

        overlay = frame.copy()

        for track_id, points in self._trajectories.items():
            if len(points) < 2:
                continue

            color = self._palette.get_color(track_id)
            display_points = self._smooth_points(points) if self._smooth else points

            for i in range(1, len(display_points)):
                if self._fade:
                    alpha = i / len(display_points)
                    thickness = max(1, int(alpha * self._line_thickness))
                    draw_color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
                else:
                    thickness = self._line_thickness
                    draw_color = color

                pt1 = (int(display_points[i - 1][0]), int(display_points[i - 1][1]))
                pt2 = (int(display_points[i][0]), int(display_points[i][1]))
                cv2.line(overlay, pt1, pt2, draw_color, thickness, cv2.LINE_AA)

            last_pt = (int(display_points[-1][0]), int(display_points[-1][1]))
            cv2.circle(overlay, last_pt, self._dot_radius + 1, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, last_pt, self._dot_radius - 1, (255, 255, 255), -1, cv2.LINE_AA)

            if self._show_direction and len(display_points) >= 2:
                p1 = display_points[-2]
                p2 = display_points[-1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = (dx * dx + dy * dy) ** 0.5
                if length > 2:
                    dx, dy = dx / length * 15, dy / length * 15
                    tip = (int(p2[0] + dx), int(p2[1] + dy))
                    cv2.arrowedLine(overlay, last_pt, tip, color, 2, cv2.LINE_AA, tipLength=0.4)

        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        return frame

    def _smooth_points(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(points) <= self._smooth_window:
            return points

        smoothed = []
        for i in range(len(points)):
            start = max(0, i - self._smooth_window // 2)
            end = min(len(points), i + self._smooth_window // 2 + 1)
            window = points[start:end]
            avg_x = sum(p[0] for p in window) / len(window)
            avg_y = sum(p[1] for p in window) / len(window)
            smoothed.append((avg_x, avg_y))

        return smoothed

    def clear(self) -> None:
        self._trajectories.clear()

    def get_trajectory(self, track_id: int) -> list[tuple[float, float]]:
        return self._trajectories.get(track_id, [])
