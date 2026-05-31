from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from src.capture.base_capture import BaseCaptureSource, CaptureBackend
from src.core.base import ComponentState
from src.core.exceptions import CaptureConnectionError, CaptureTimeoutError
from src.core.registry import capture_registry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@capture_registry.register("rtsp")
class RTSPCapture(BaseCaptureSource):

    def __init__(
        self,
        rtsp_url: str,
        transport: str = "tcp",
        timeout: float = 10.0,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0,
        use_threading: bool = True,
        buffer_size: int = 1,
        name: str | None = None,
    ) -> None:
        super().__init__(
            source=rtsp_url,
            name=name or "RTSP-Stream",
        )
        self._rtsp_url = rtsp_url
        self._transport = transport
        self._timeout = timeout
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._use_threading = use_threading
        self._buffer_size = buffer_size
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._running = False
        self._connection_lost_count: int = 0

    @property
    def rtsp_url(self) -> str:
        return self._rtsp_url

    @property
    def connection_lost_count(self) -> int:
        return self._connection_lost_count

    def initialize(self) -> None:
        self.set_state(ComponentState.INITIALIZING)
        logger.info("initializing_rtsp", url=self._rtsp_url, transport=self._transport)

        self._connect()

        if self._use_threading:
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

        self.set_state(ComponentState.READY)

    def _connect(self) -> None:
        for attempt in range(self._reconnect_attempts):
            try:
                if self._transport == "tcp":
                    env_options = f"rtsp_transport;tcp"
                    self._cap = cv2.VideoCapture(self._rtsp_url, cv2.CAP_FFMPEG)
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
                else:
                    self._cap = cv2.VideoCapture(self._rtsp_url)
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

                if self._cap.isOpened():
                    self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

                    logger.info(
                        "rtsp_connected",
                        url=self._rtsp_url,
                        resolution=f"{self._width}x{self._height}",
                        fps=self._fps,
                        attempt=attempt + 1,
                    )
                    return

            except Exception as e:
                logger.warning(
                    "rtsp_connection_attempt_failed",
                    attempt=attempt + 1,
                    max_attempts=self._reconnect_attempts,
                    error=str(e),
                )

            if attempt < self._reconnect_attempts - 1:
                time.sleep(self._reconnect_delay)

        self.set_state(ComponentState.ERROR)
        raise CaptureConnectionError(
            source=self._rtsp_url,
            reason=f"Failed after {self._reconnect_attempts} attempts",
        )

    def _capture_loop(self) -> None:
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._connection_lost_count += 1
                logger.warning("rtsp_connection_lost", count=self._connection_lost_count)
                try:
                    self._connect()
                except Exception:
                    time.sleep(self._reconnect_delay)
                    continue

            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._latest_frame = frame
                    self._frame_count += 1
            else:
                self._dropped_frames += 1
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None

    def read(self) -> tuple[bool, np.ndarray | None]:
        self._total_frames_read += 1

        if self._use_threading:
            with self._lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
                    return True, frame
                return False, None

        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if ret:
            self._frame_count += 1
            return True, frame

        self._dropped_frames += 1
        return False, None

    def is_opened(self) -> bool:
        if self._use_threading:
            return self._running
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("rtsp_released", url=self._rtsp_url)

    def get_total_frames(self) -> int:
        return -1

    def get_current_position(self) -> int:
        return self._frame_count

    def seek(self, frame_number: int) -> bool:
        return False

    def shutdown(self) -> None:
        self.release()
        self.set_state(ComponentState.STOPPED)
