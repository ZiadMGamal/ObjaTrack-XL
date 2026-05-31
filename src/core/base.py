from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np


class ComponentState(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class BaseComponent(ABC):

    def __init__(self, name: str | None = None) -> None:
        self._name = name or self.__class__.__name__
        self._state = ComponentState.UNINITIALIZED
        self._metadata: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> ComponentState:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state == ComponentState.READY

    @property
    def is_running(self) -> bool:
        return self._state == ComponentState.RUNNING

    def set_state(self, state: ComponentState) -> None:
        self._state = state

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "state": self._state.value,
            "class": self.__class__.__name__,
            **self._metadata,
        }

    @abstractmethod
    def initialize(self) -> None:
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r}, state={self._state.value!r})"


class BaseCapture(BaseComponent):

    def __init__(self, source: str | int, name: str | None = None) -> None:
        super().__init__(name=name)
        self._source = source
        self._frame_count: int = 0
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 0.0

    @property
    def source(self) -> str | int:
        return self._source

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def fps(self) -> float:
        return self._fps

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        ...

    @abstractmethod
    def is_opened(self) -> bool:
        ...

    @abstractmethod
    def release(self) -> None:
        ...

    def __enter__(self) -> BaseCapture:
        self.initialize()
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        self.release()
        self.shutdown()


class DetectionResult:

    __slots__ = ("boxes", "scores", "class_ids", "class_names", "masks", "keypoints", "frame_id", "timestamp")

    def __init__(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        class_names: list[str] | None = None,
        masks: np.ndarray | None = None,
        keypoints: np.ndarray | None = None,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> None:
        self.boxes = boxes
        self.scores = scores
        self.class_ids = class_ids
        self.class_names = class_names
        self.masks = masks
        self.keypoints = keypoints
        self.frame_id = frame_id
        self.timestamp = timestamp

    @property
    def num_detections(self) -> int:
        return len(self.boxes)

    def filter_by_confidence(self, threshold: float) -> DetectionResult:
        mask = self.scores >= threshold
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            class_ids=self.class_ids[mask],
            class_names=[n for n, m in zip(self.class_names or [], mask) if m] or None,
            masks=self.masks[mask] if self.masks is not None else None,
            keypoints=self.keypoints[mask] if self.keypoints is not None else None,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
        )

    def filter_by_classes(self, class_ids: list[int]) -> DetectionResult:
        mask = np.isin(self.class_ids, class_ids)
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            class_ids=self.class_ids[mask],
            class_names=[n for n, m in zip(self.class_names or [], mask) if m] or None,
            masks=self.masks[mask] if self.masks is not None else None,
            keypoints=self.keypoints[mask] if self.keypoints is not None else None,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "num_detections": self.num_detections,
            "detections": [
                {
                    "box": self.boxes[i].tolist(),
                    "score": float(self.scores[i]),
                    "class_id": int(self.class_ids[i]),
                    "class_name": self.class_names[i] if self.class_names else None,
                }
                for i in range(self.num_detections)
            ],
        }

    def __len__(self) -> int:
        return self.num_detections

    def __repr__(self) -> str:
        return f"DetectionResult(num_detections={self.num_detections}, frame_id={self.frame_id})"


class BaseDetector(BaseComponent):

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._input_size: tuple[int, int] = (640, 640)
        self._class_names: dict[int, str] = {}

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        self._confidence_threshold = max(0.0, min(1.0, value))

    @property
    def iou_threshold(self) -> float:
        return self._iou_threshold

    @iou_threshold.setter
    def iou_threshold(self, value: float) -> None:
        self._iou_threshold = max(0.0, min(1.0, value))

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        ...

    @abstractmethod
    def warmup(self, iterations: int = 10) -> None:
        ...

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def postprocess(self, raw_output: Any, original_shape: tuple[int, ...]) -> DetectionResult:
        raise NotImplementedError


class BaseTracker(BaseComponent):

    def __init__(self, max_age: int = 30, min_hits: int = 3, name: str | None = None) -> None:
        super().__init__(name=name)
        self._max_age = max_age
        self._min_hits = min_hits
        self._frame_count: int = 0
        self._active_tracks: int = 0
        self._total_tracks: int = 0

    @property
    def max_age(self) -> int:
        return self._max_age

    @property
    def min_hits(self) -> int:
        return self._min_hits

    @property
    def active_tracks(self) -> int:
        return self._active_tracks

    @property
    def total_tracks(self) -> int:
        return self._total_tracks

    @abstractmethod
    def update(self, detections: DetectionResult, frame: np.ndarray | None = None) -> list[Any]:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    def get_statistics(self) -> dict[str, Any]:
        return {
            "active_tracks": self._active_tracks,
            "total_tracks": self._total_tracks,
            "frame_count": self._frame_count,
            "max_age": self._max_age,
            "min_hits": self._min_hits,
        }


class BaseOptimizer(BaseComponent):

    def __init__(self, model_path: str, output_dir: str, name: str | None = None) -> None:
        super().__init__(name=name)
        self._model_path = model_path
        self._output_dir = output_dir

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @abstractmethod
    def optimize(self) -> str:
        ...

    @abstractmethod
    def validate(self, original_path: str, optimized_path: str) -> dict[str, Any]:
        ...
