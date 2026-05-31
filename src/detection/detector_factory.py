from __future__ import annotations

from typing import Any

from src.config.settings import ModelSettings
from src.core.exceptions import DetectionError
from src.detection.base_detector import BaseObjectDetector
from src.detection.onnx_detector import ONNXDetector
from src.detection.yolo_detector import YOLODetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DetectorFactory:
    _detector_map: dict[str, type[BaseObjectDetector]] = {
        "yolo": YOLODetector,
        "pytorch": YOLODetector,
        "onnx": ONNXDetector,
        "onnxruntime": ONNXDetector,
    }

    @classmethod
    def create(
        cls,
        detector_type: str,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None,
        input_size: tuple[int, int] = (640, 640),
        **kwargs: Any,
    ) -> BaseObjectDetector:
        detector_type = detector_type.lower()

        if detector_type not in cls._detector_map:
            available = list(cls._detector_map.keys())
            raise DetectionError(f"Unknown detector type: {detector_type}. Available: {available}")

        detector_cls = cls._detector_map[detector_type]

        logger.info(
            "creating_detector",
            type=detector_type,
            model=model_path,
            confidence=confidence_threshold,
        )

        detector = detector_cls(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
            input_size=input_size,
            **kwargs,
        )

        return detector

    @classmethod
    def create_from_settings(cls, settings: ModelSettings, **kwargs: Any) -> BaseObjectDetector:
        model_name = settings.name
        if model_name.endswith(".onnx"):
            detector_type = "onnx"
        elif model_name.endswith(".engine") or model_name.endswith(".trt"):
            detector_type = "tensorrt"
        else:
            detector_type = "yolo"

        return cls.create(
            detector_type=detector_type,
            model_path=model_name,
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
            classes=settings.classes,
            input_size=settings.input_size,
            half_precision=settings.half_precision,
            **kwargs,
        )

    @classmethod
    def register_detector(cls, name: str, detector_cls: type[BaseObjectDetector]) -> None:
        cls._detector_map[name.lower()] = detector_cls
        logger.info("detector_registered", name=name)

    @classmethod
    def available_detectors(cls) -> list[str]:
        return list(cls._detector_map.keys())
