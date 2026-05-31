from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoWriter:
    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        codec: str = "mp4v",
        width: int = 0,
        height: int = 0,
    ) -> None:
        self._output_path = Path(output_path)
        self._fps = fps
        self._codec = codec
        self._width = width
        self._height = height
        self._writer: cv2.VideoWriter | None = None
        self._frame_count: int = 0
        self._initialized = False

    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_opened(self) -> bool:
        return self._writer is not None and self._writer.isOpened()

    def initialize(self, width: int | None = None, height: int | None = None) -> None:
        if width:
            self._width = width
        if height:
            self._height = height

        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        self._writer = cv2.VideoWriter(
            str(self._output_path),
            fourcc,
            self._fps,
            (self._width, self._height),
        )

        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self._output_path}")

        self._initialized = True
        logger.info(
            "video_writer_initialized",
            path=str(self._output_path),
            resolution=f"{self._width}x{self._height}",
            fps=self._fps,
        )

    def write(self, frame: np.ndarray) -> None:
        if not self._initialized:
            h, w = frame.shape[:2]
            self.initialize(width=w, height=h)

        if self._writer is None:
            return

        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            frame = cv2.resize(frame, (self._width, self._height))

        self._writer.write(frame)
        self._frame_count += 1

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info(
                "video_writer_released",
                path=str(self._output_path),
                frames_written=self._frame_count,
            )

    def get_info(self) -> dict[str, Any]:
        return {
            "path": str(self._output_path),
            "fps": self._fps,
            "codec": self._codec,
            "resolution": f"{self._width}x{self._height}",
            "frames_written": self._frame_count,
            "duration_seconds": round(self._frame_count / self._fps, 2) if self._fps > 0 else 0,
            "is_opened": self.is_opened,
        }

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        self.release()
