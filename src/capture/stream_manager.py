from __future__ import annotations

import threading
from typing import Any

import numpy as np

from src.capture.base_capture import BaseCaptureSource
from src.capture.camera_capture import CameraCapture
from src.capture.file_capture import FileCapture
from src.capture.rtsp_capture import RTSPCapture
from src.capture.frame_buffer import FrameBuffer
from src.config.settings import CaptureSettings
from src.core.base import ComponentState
from src.core.exceptions import CaptureError, ConfigurationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StreamManager:

    def __init__(self, settings: CaptureSettings | None = None) -> None:
        self._settings = settings or CaptureSettings()
        self._sources: dict[str, BaseCaptureSource] = {}
        self._buffers: dict[str, FrameBuffer] = {}
        self._active_source: str | None = None
        self._lock = threading.Lock()

    @property
    def active_source_name(self) -> str | None:
        return self._active_source

    @property
    def active_source(self) -> BaseCaptureSource | None:
        if self._active_source is None:
            return None
        return self._sources.get(self._active_source)

    @property
    def source_count(self) -> int:
        return len(self._sources)

    def create_source_from_settings(self, settings: CaptureSettings | None = None) -> BaseCaptureSource:
        config = settings or self._settings
        source_type = config.type.lower()

        if source_type == "file":
            source = FileCapture(
                file_path=config.path,
                loop=config.loop,
                frame_skip=config.frame_skip,
            )
        elif source_type == "webcam":
            source = CameraCapture(
                camera_index=config.webcam_index,
                width=config.width,
                height=config.height,
                fps=config.fps,
            )
        elif source_type == "rtsp":
            source = RTSPCapture(
                rtsp_url=config.rtsp_url,
                reconnect_attempts=config.reconnect_attempts,
                reconnect_delay=config.reconnect_delay,
            )
        else:
            raise ConfigurationError(f"Unknown source type: {source_type}")

        self.add_source(source.name, source)
        return source

    def add_source(
        self,
        name: str,
        source: BaseCaptureSource,
        buffer_size: int = 128,
    ) -> None:
        with self._lock:
            if name in self._sources:
                raise CaptureError(f"Source '{name}' already exists")

            self._sources[name] = source
            self._buffers[name] = FrameBuffer(max_size=buffer_size, name=name)

            if self._active_source is None:
                self._active_source = name

            logger.info("source_added", name=name, type=type(source).__name__)

    def remove_source(self, name: str) -> None:
        with self._lock:
            if name not in self._sources:
                return

            source = self._sources[name]
            if source.is_opened():
                source.release()

            del self._sources[name]
            if name in self._buffers:
                self._buffers[name].close()
                del self._buffers[name]

            if self._active_source == name:
                self._active_source = next(iter(self._sources), None)

            logger.info("source_removed", name=name)

    def switch_source(self, name: str) -> None:
        with self._lock:
            if name not in self._sources:
                raise CaptureError(f"Source '{name}' not found")
            self._active_source = name
            logger.info("source_switched", name=name)

    def initialize_active(self) -> None:
        source = self.active_source
        if source is None:
            raise CaptureError("No active source configured")

        if not source.is_ready:
            source.initialize()

    def read_active(self) -> tuple[bool, np.ndarray | None]:
        source = self.active_source
        if source is None:
            return False, None
        return source.read()

    def release_all(self) -> None:
        with self._lock:
            for name, source in self._sources.items():
                try:
                    if source.is_opened():
                        source.release()
                except Exception as e:
                    logger.error("source_release_error", name=name, error=str(e))

            for buffer in self._buffers.values():
                buffer.close()

            logger.info("all_sources_released", count=len(self._sources))

    def get_source(self, name: str) -> BaseCaptureSource | None:
        return self._sources.get(name)

    def list_sources(self) -> list[dict[str, Any]]:
        result = []
        for name, source in self._sources.items():
            info = source.get_capture_info()
            info["name"] = name
            info["is_active"] = name == self._active_source
            result.append(info)
        return result

    def get_buffer_stats(self) -> dict[str, Any]:
        return {name: buf.get_statistics() for name, buf in self._buffers.items()}

    def __enter__(self) -> StreamManager:
        self.initialize_active()
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        self.release_all()
