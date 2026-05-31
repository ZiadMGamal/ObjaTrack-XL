from __future__ import annotations

import cv2
import numpy as np

from src.core.base import DetectionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionHeatmap:
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        decay: float = 0.95,
        intensity: float = 2.0,
        blur_size: int = 51,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.4,
    ) -> None:
        self._width = width
        self._height = height
        self._decay = decay
        self._intensity = intensity
        self._blur_size = blur_size
        self._colormap = colormap
        self._alpha = alpha
        self._accumulator = np.zeros((height, width), dtype=np.float32)
        self._frame_count: int = 0

    def update(self, detections: DetectionResult) -> None:
        self._accumulator *= self._decay
        self._frame_count += 1

        for i in range(detections.num_detections):
            box = detections.boxes[i].astype(int)
            x1 = max(0, min(box[0], self._width - 1))
            y1 = max(0, min(box[1], self._height - 1))
            x2 = max(0, min(box[2], self._width - 1))
            y2 = max(0, min(box[3], self._height - 1))

            if x2 > x1 and y2 > y1:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                sigma_x = max(w // 4, 5)
                sigma_y = max(h // 4, 5)

                y_grid, x_grid = np.ogrid[
                    max(0, cy - 3 * sigma_y) : min(self._height, cy + 3 * sigma_y),
                    max(0, cx - 3 * sigma_x) : min(self._width, cx + 3 * sigma_x),
                ]
                gaussian = (
                    np.exp(-((x_grid - cx) ** 2 / (2 * sigma_x**2) + (y_grid - cy) ** 2 / (2 * sigma_y**2)))
                    * self._intensity
                )

                y_start = max(0, cy - 3 * sigma_y)
                x_start = max(0, cx - 3 * sigma_x)
                self._accumulator[
                    y_start : y_start + gaussian.shape[0],
                    x_start : x_start + gaussian.shape[1],
                ] += gaussian.astype(np.float32)

    def render(self, frame: np.ndarray) -> np.ndarray:
        normalized = np.clip(self._accumulator / max(self._accumulator.max(), 1.0), 0, 1)
        blurred = cv2.GaussianBlur(normalized, (self._blur_size, self._blur_size), 0)
        heatmap_uint8 = (blurred * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, self._colormap)

        if heatmap_colored.shape[:2] != frame.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))

        mask = blurred > 0.05
        mask_3d = np.stack([mask] * 3, axis=-1)

        output = frame.copy()
        blended = cv2.addWeighted(frame, 1 - self._alpha, heatmap_colored, self._alpha, 0)
        output[mask_3d] = blended[mask_3d]

        return output

    def get_raw_heatmap(self) -> np.ndarray:
        return self._accumulator.copy()

    def reset(self) -> None:
        self._accumulator = np.zeros((self._height, self._width), dtype=np.float32)
        self._frame_count = 0

    def resize(self, width: int, height: int) -> None:
        self._accumulator = cv2.resize(self._accumulator, (width, height))
        self._width = width
        self._height = height
