from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.capture.base_capture import BaseCaptureSource
from src.core.base import ComponentState
from src.core.exceptions import CaptureConnectionError
from src.core.registry import capture_registry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@capture_registry.register("file")
class FileCapture(BaseCaptureSource):
    def __init__(
        self,
        file_path: str,
        loop: bool = True,
        start_frame: int = 0,
        end_frame: int | None = None,
        frame_skip: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(
            source=file_path,
            name=name or f"File-{Path(file_path).stem}",
        )
        self._file_path = Path(file_path)
        self._loop = loop
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._frame_skip = frame_skip
        self._cap: cv2.VideoCapture | None = None
        self._total_video_frames: int = 0
        self._skip_counter: int = 0

    @property
    def file_path(self) -> Path:
        return self._file_path

    @property
    def loop(self) -> bool:
        return self._loop

    @property
    def total_video_frames(self) -> int:
        return self._total_video_frames

    @property
    def progress(self) -> float:
        if self._total_video_frames == 0:
            return 0.0
        return self._frame_count / self._total_video_frames

    def initialize(self) -> None:
        self.set_state(ComponentState.INITIALIZING)

        if not self._file_path.exists():
            self.set_state(ComponentState.ERROR)
            raise CaptureConnectionError(
                source=str(self._file_path),
                reason=f"Video file not found: {self._file_path}",
            )

        self._cap = cv2.VideoCapture(str(self._file_path))

        if not self._cap.isOpened():
            self.set_state(ComponentState.ERROR)
            raise CaptureConnectionError(
                source=str(self._file_path),
                reason="Failed to open video file",
            )

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_video_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self._start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)

        if self._end_frame is not None:
            self._end_frame = min(self._end_frame, self._total_video_frames)

        self.set_state(ComponentState.READY)
        logger.info(
            "file_capture_initialized",
            path=str(self._file_path),
            resolution=f"{self._width}x{self._height}",
            fps=self._fps,
            total_frames=self._total_video_frames,
            loop=self._loop,
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None or not self._cap.isOpened():
            return False, None

        if self._end_frame is not None:
            current = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current >= self._end_frame:
                if self._loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)
                else:
                    return False, None

        for _ in range(self._frame_skip):
            self._cap.grab()
            self._skip_counter += 1

        ret, frame = self._cap.read()
        self._total_frames_read += 1

        if not ret:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)
                ret, frame = self._cap.read()
                if not ret:
                    self._dropped_frames += 1
                    return False, None
            else:
                return False, None

        self._frame_count += 1
        return True, frame

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("file_capture_released", path=str(self._file_path))

    def get_total_frames(self) -> int:
        return self._total_video_frames

    def get_current_position(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    def seek(self, frame_number: int) -> bool:
        if self._cap is None:
            return False
        frame_number = max(0, min(frame_number, self._total_video_frames - 1))
        return self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def get_duration_seconds(self) -> float:
        if self._fps == 0:
            return 0.0
        return self._total_video_frames / self._fps

    def get_file_info(self) -> dict:
        return {
            "path": str(self._file_path),
            "filename": self._file_path.name,
            "size_mb": round(self._file_path.stat().st_size / (1024 * 1024), 2) if self._file_path.exists() else 0,
            "duration_seconds": round(self.get_duration_seconds(), 2),
            "total_frames": self._total_video_frames,
            "resolution": f"{self._width}x{self._height}",
            "fps": self._fps,
            "loop": self._loop,
        }

    def shutdown(self) -> None:
        self.release()
        self.set_state(ComponentState.STOPPED)
