from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from src.core.base import ComponentState, DetectionResult
from src.core.exceptions import DetectionError, ModelLoadError
from src.core.registry import detector_registry
from src.detection.base_detector import BaseObjectDetector
from src.detection.nms import non_max_suppression
from src.detection.preprocessing import DetectionPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


@detector_registry.register("onnx")
class ONNXDetector(BaseObjectDetector):
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None,
        max_detections: int = 300,
        input_size: tuple[int, int] = (640, 640),
        providers: list[str] | None = None,
        intra_op_threads: int = 4,
        inter_op_threads: int = 2,
        class_names: dict[int, str] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
            max_detections=max_detections,
            input_size=input_size,
            name=name or "ONNXDetector",
        )
        self._providers = providers or ["CPUExecutionProvider"]
        self._intra_op_threads = intra_op_threads
        self._inter_op_threads = inter_op_threads
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._output_names: list[str] = []
        self._preprocessor = DetectionPreprocessor(input_size=input_size)

        if class_names:
            self._class_names = class_names

    @property
    def session(self) -> ort.InferenceSession | None:
        return self._session

    @property
    def providers(self) -> list[str]:
        return self._providers

    def initialize(self) -> None:
        self.set_state(ComponentState.INITIALIZING)
        logger.info("initializing_onnx", model=self._model_path, providers=self._providers)

        if not Path(self._model_path).exists():
            self.set_state(ComponentState.ERROR)
            raise ModelLoadError(self._model_path, "ONNX model file not found")

        try:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = self._intra_op_threads
            session_options.inter_op_num_threads = self._inter_op_threads
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True

            self._session = ort.InferenceSession(
                self._model_path,
                sess_options=session_options,
                providers=self._providers,
            )
        except Exception as e:
            self.set_state(ComponentState.ERROR)
            raise ModelLoadError(self._model_path, str(e))

        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        input_shape = self._session.get_inputs()[0].shape
        if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            self._input_size = (input_shape[2], input_shape[3])

        active_providers = self._session.get_providers()

        self.set_state(ComponentState.READY)
        logger.info(
            "onnx_initialized",
            model=self._model_path,
            input_name=self._input_name,
            input_size=self._input_size,
            active_providers=active_providers,
        )

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if self._session is None:
            raise DetectionError("ONNX session not initialized")

        self.set_state(ComponentState.RUNNING)
        timestamp = time.time()
        original_shape = frame.shape[:2]

        input_tensor, ratio, padding = self._preprocessor.preprocess(frame)

        try:
            outputs = self._session.run(
                self._output_names,
                {self._input_name: input_tensor},
            )
        except Exception as e:
            self.set_state(ComponentState.READY)
            raise DetectionError(f"ONNX inference failed: {e}")

        detection_result = self._postprocess(outputs[0], original_shape, ratio, padding, timestamp)

        detection_result = self.filter_classes(detection_result)
        self._inference_count += 1
        self._total_detections += detection_result.num_detections
        self.set_state(ComponentState.READY)

        return detection_result

    def _postprocess(
        self,
        output: np.ndarray,
        original_shape: tuple[int, int],
        ratio: float,
        padding: tuple[float, float],
        timestamp: float,
    ) -> DetectionResult:
        if output.ndim == 3:
            output = output[0]

        if output.shape[0] < output.shape[1]:
            output = output.T

        boxes_xywh = output[:, :4]
        scores_all = output[:, 4:]

        if scores_all.shape[1] == 1:
            scores = scores_all[:, 0]
            class_ids = np.zeros(len(scores), dtype=np.int32)
        else:
            scores = np.max(scores_all, axis=1)
            class_ids = np.argmax(scores_all, axis=1)

        mask = scores >= self._confidence_threshold
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_names=[],
                frame_id=self._inference_count,
                timestamp=timestamp,
            )

        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - padding[0]) / ratio
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - padding[1]) / ratio

        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, original_shape[1])
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, original_shape[0])

        keep = non_max_suppression(boxes_xyxy, scores, self._iou_threshold)
        if len(keep) > self._max_detections:
            keep = keep[: self._max_detections]

        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        class_ids = class_ids[keep].astype(np.int32)
        class_names_list = [self._class_names.get(int(c), "unknown") for c in class_ids]

        return DetectionResult(
            boxes=boxes_xyxy.astype(np.float32),
            scores=scores.astype(np.float32),
            class_ids=class_ids,
            class_names=class_names_list,
            frame_id=self._inference_count,
            timestamp=timestamp,
        )

    def warmup(self, iterations: int = 10) -> None:
        if self._session is None:
            raise DetectionError("ONNX session not initialized for warmup")

        logger.info("onnx_warmup_start", iterations=iterations)
        dummy = np.random.randn(1, 3, *self._input_size).astype(np.float32)

        for _ in range(iterations):
            self._session.run(self._output_names, {self._input_name: dummy})

        logger.info("onnx_warmup_complete", iterations=iterations)

    def get_model_metadata(self) -> dict[str, Any]:
        if self._session is None:
            return {}

        metadata = {}
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()

        metadata["inputs"] = [{"name": i.name, "shape": i.shape, "type": i.type} for i in inputs]
        metadata["outputs"] = [{"name": o.name, "shape": o.shape, "type": o.type} for o in outputs]
        metadata["providers"] = self._session.get_providers()

        return metadata

    def shutdown(self) -> None:
        self._session = None
        self.set_state(ComponentState.STOPPED)
        logger.info("onnx_shutdown")
