from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Any

import numpy as np

from src.core.base import BaseCapture
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CaptureBackend(Enum):
    OPENCV = "opencv"
    FFMPEG = "ffmpeg"
    GSTREAMER = "gstreamer"


class BaseCaptureSource(BaseCapture):
    def __init__(
        self,
        source: str | int,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        backend: CaptureBackend = CaptureBackend.OPENCV,
        name: str | None = None,
    ) -> None:
        super().__init__(source=source, name=name)
        self._target_width = width
        self._target_height = height
        self._target_fps = fps
        self._backend = backend
        self._total_frames_read: int = 0
        self._dropped_frames: int = 0

    @property
    def target_width(self) -> int:
        return self._target_width

    @property
    def target_height(self) -> int:
        return self._target_height

    @property
    def target_fps(self) -> float:
        return self._target_fps

    @property
    def total_frames_read(self) -> int:
        return self._total_frames_read

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def drop_rate(self) -> float:
        if self._total_frames_read == 0:
            return 0.0
        return self._dropped_frames / self._total_frames_read

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]: ...

    @abstractmethod
    def is_opened(self) -> bool: ...

    @abstractmethod
    def release(self) -> None: ...

    @abstractmethod
    def get_total_frames(self) -> int: ...

    @abstractmethod
    def get_current_position(self) -> int: ...

    @abstractmethod
    def seek(self, frame_number: int) -> bool: ...

    def get_capture_info(self) -> dict[str, Any]:
        return {
            "source": str(self._source),
            "resolution": f"{self._width}x{self._height}",
            "fps": self._fps,
            "total_frames_read": self._total_frames_read,
            "dropped_frames": self._dropped_frames,
            "drop_rate": f"{self.drop_rate:.2%}",
            "state": self._state.value,
            "backend": self._backend.value,
        }
