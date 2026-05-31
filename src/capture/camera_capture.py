from __future__ import annotations

import cv2
import numpy as np

from src.capture.base_capture import BaseCaptureSource, CaptureBackend
from src.core.base import ComponentState
from src.core.exceptions import CaptureConnectionError
from src.core.registry import capture_registry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@capture_registry.register("webcam")
class CameraCapture(BaseCaptureSource):
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        auto_exposure: bool = True,
        backend: CaptureBackend = CaptureBackend.OPENCV,
        name: str | None = None,
    ) -> None:
        super().__init__(
            source=camera_index,
            width=width,
            height=height,
            fps=fps,
            backend=backend,
            name=name or f"Camera-{camera_index}",
        )
        self._camera_index = camera_index
        self._auto_exposure = auto_exposure
        self._cap: cv2.VideoCapture | None = None

    @property
    def camera_index(self) -> int:
        return self._camera_index

    def initialize(self) -> None:
        self.set_state(ComponentState.INITIALIZING)
        logger.info("initializing_camera", camera_index=self._camera_index)

        backend_map = {
            CaptureBackend.OPENCV: cv2.CAP_ANY,
            CaptureBackend.GSTREAMER: cv2.CAP_GSTREAMER,
        }
        api = backend_map.get(self._backend, cv2.CAP_ANY)

        self._cap = cv2.VideoCapture(self._camera_index, api)

        if not self._cap.isOpened():
            self.set_state(ComponentState.ERROR)
            raise CaptureConnectionError(
                source=f"camera:{self._camera_index}",
                reason="Failed to open camera device",
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if self._auto_exposure:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or self._target_fps

        self.set_state(ComponentState.READY)
        logger.info(
            "camera_initialized",
            camera_index=self._camera_index,
            resolution=f"{self._width}x{self._height}",
            fps=self._fps,
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        self._total_frames_read += 1

        if not ret:
            self._dropped_frames += 1
            return False, None

        self._frame_count += 1
        return True, frame

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("camera_released", camera_index=self._camera_index)

    def get_total_frames(self) -> int:
        return -1

    def get_current_position(self) -> int:
        return self._frame_count

    def seek(self, frame_number: int) -> bool:
        return False

    def shutdown(self) -> None:
        self.release()
        self.set_state(ComponentState.STOPPED)
