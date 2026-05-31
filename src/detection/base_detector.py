from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from src.core.base import BaseDetector, DetectionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseObjectDetector(BaseDetector):
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None,
        max_detections: int = 300,
        input_size: tuple[int, int] = (640, 640),
        half_precision: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            name=name,
        )
        self._target_classes = classes
        self._max_detections = max_detections
        self._input_size = input_size
        self._half_precision = half_precision
        self._inference_count: int = 0
        self._total_detections: int = 0

    @property
    def target_classes(self) -> list[int] | None:
        return self._target_classes

    @property
    def max_detections(self) -> int:
        return self._max_detections

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    @property
    def inference_count(self) -> int:
        return self._inference_count

    @property
    def average_detections(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return self._total_detections / self._inference_count

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult: ...

    @abstractmethod
    def warmup(self, iterations: int = 10) -> None: ...

    def filter_classes(self, result: DetectionResult) -> DetectionResult:
        if self._target_classes is None:
            return result
        return result.filter_by_classes(self._target_classes)

    def get_detector_info(self) -> dict[str, Any]:
        return {
            "model_path": self._model_path,
            "confidence_threshold": self._confidence_threshold,
            "iou_threshold": self._iou_threshold,
            "input_size": self._input_size,
            "target_classes": self._target_classes,
            "max_detections": self._max_detections,
            "half_precision": self._half_precision,
            "inference_count": self._inference_count,
            "average_detections": round(self.average_detections, 2),
            "class_names": self._class_names,
        }
