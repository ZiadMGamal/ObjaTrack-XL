from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from src.core.base import ComponentState, DetectionResult
from src.core.exceptions import DetectionError, ModelLoadError, ModelNotFoundError
from src.core.registry import detector_registry
from src.detection.base_detector import BaseObjectDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


@detector_registry.register("yolo")
class YOLODetector(BaseObjectDetector):

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None,
        max_detections: int = 300,
        input_size: tuple[int, int] = (640, 640),
        half_precision: bool = False,
        device: str = "auto",
        name: str | None = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
            max_detections=max_detections,
            input_size=input_size,
            half_precision=half_precision,
            name=name or "YOLODetector",
        )
        self._device = device
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO | None:
        return self._model

    def initialize(self) -> None:
        self.set_state(ComponentState.INITIALIZING)
        logger.info("initializing_yolo", model=self._model_path, device=self._device)

        try:
            self._model = YOLO(self._model_path)
        except Exception as e:
            self.set_state(ComponentState.ERROR)
            raise ModelLoadError(self._model_path, str(e))

        if hasattr(self._model, "names"):
            self._class_names = dict(self._model.names)

        self.set_state(ComponentState.READY)
        logger.info(
            "yolo_initialized",
            model=self._model_path,
            classes=len(self._class_names),
        )

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if self._model is None:
            raise DetectionError("Model not initialized", model_name=self._model_path)

        self.set_state(ComponentState.RUNNING)
        timestamp = time.time()

        try:
            results = self._model.predict(
                source=frame,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                max_det=self._max_detections,
                imgsz=self._input_size[0],
                half=self._half_precision,
                device=self._device if self._device != "auto" else None,
                verbose=False,
            )
        except Exception as e:
            self.set_state(ComponentState.READY)
            raise DetectionError(f"Inference failed: {e}", model_name=self._model_path)

        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            class_names_list = [self._class_names.get(int(c), "unknown") for c in class_ids]
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            class_ids = np.empty((0,), dtype=np.int32)
            class_names_list = []

        detection_result = DetectionResult(
            boxes=xyxy,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names_list,
            frame_id=self._inference_count,
            timestamp=timestamp,
        )

        detection_result = self.filter_classes(detection_result)
        self._inference_count += 1
        self._total_detections += detection_result.num_detections
        self.set_state(ComponentState.READY)

        return detection_result

    def warmup(self, iterations: int = 10) -> None:
        if self._model is None:
            raise DetectionError("Model not initialized for warmup")

        logger.info("yolo_warmup_start", iterations=iterations)
        dummy = np.zeros((self._input_size[0], self._input_size[1], 3), dtype=np.uint8)

        for i in range(iterations):
            self._model.predict(
                source=dummy,
                conf=self._confidence_threshold,
                imgsz=self._input_size[0],
                verbose=False,
            )

        logger.info("yolo_warmup_complete", iterations=iterations)

    def export_to_onnx(
        self,
        output_path: str | None = None,
        opset: int = 17,
        simplify: bool = True,
        dynamic: bool = True,
        half: bool = False,
    ) -> str:
        if self._model is None:
            raise DetectionError("Model not initialized for export")

        export_args: dict[str, Any] = {
            "format": "onnx",
            "opset": opset,
            "simplify": simplify,
            "dynamic": dynamic,
            "half": half,
        }

        logger.info("exporting_to_onnx", **export_args)
        result = self._model.export(**export_args)
        logger.info("onnx_export_complete", path=str(result))
        return str(result)

    def shutdown(self) -> None:
        self._model = None
        self.set_state(ComponentState.STOPPED)
        logger.info("yolo_shutdown")
