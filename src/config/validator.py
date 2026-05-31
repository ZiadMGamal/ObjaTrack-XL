from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config.settings import Settings
from src.core.exceptions import ConfigValidationError, ConfigurationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigValidator:

    SUPPORTED_MODELS = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}
    SUPPORTED_TRACKERS = {"botsort", "bytetrack", "sort", "deepsort"}
    SUPPORTED_SOURCES = {"file", "webcam", "rtsp"}
    SUPPORTED_EXPORT_FORMATS = {"onnx", "tensorrt", "openvino"}
    SUPPORTED_DEVICES = {"auto", "cpu", "cuda", "mps"}

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._errors: list[str] = []
        self._warnings: list[str] = []

    @property
    def errors(self) -> list[str]:
        return self._errors

    @property
    def warnings(self) -> list[str]:
        return self._warnings

    @property
    def is_valid(self) -> bool:
        return len(self._errors) == 0

    def validate(self) -> bool:
        self._errors.clear()
        self._warnings.clear()

        self._validate_device()
        self._validate_model()
        self._validate_source()
        self._validate_tracker()
        self._validate_output()
        self._validate_thresholds()

        if self._errors:
            for error in self._errors:
                logger.error("config_validation_error", error=error)
        if self._warnings:
            for warning in self._warnings:
                logger.warning("config_validation_warning", warning=warning)

        return self.is_valid

    def validate_strict(self) -> None:
        if not self.validate():
            raise ConfigurationError(
                f"Configuration validation failed with {len(self._errors)} error(s): "
                + "; ".join(self._errors)
            )

    def _validate_device(self) -> None:
        device = self._settings.device.lower()
        if device not in self.SUPPORTED_DEVICES:
            self._errors.append(
                f"Unsupported device '{device}'. Supported: {self.SUPPORTED_DEVICES}"
            )

        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    self._warnings.append("CUDA device specified but CUDA is not available. Falling back to CPU.")
            except ImportError:
                self._errors.append("PyTorch not installed but CUDA device specified")

    def _validate_model(self) -> None:
        model = self._settings.model
        model_path = Path(model.name)

        if model_path.suffix == ".pt" and model.name not in self.SUPPORTED_MODELS:
            self._warnings.append(
                f"Model '{model.name}' is not a standard YOLOv8 variant. Ensure it exists."
            )

        if model.confidence_threshold >= model.iou_threshold:
            self._warnings.append(
                "Confidence threshold >= IOU threshold. This may reduce detection quality."
            )

        if model.input_size[0] < 320 or model.input_size[1] < 320:
            self._warnings.append("Input size below 320 may significantly reduce detection accuracy.")

    def _validate_source(self) -> None:
        source = self._settings.source

        if source.type not in self.SUPPORTED_SOURCES:
            self._errors.append(
                f"Unsupported source type '{source.type}'. Supported: {self.SUPPORTED_SOURCES}"
            )

        if source.type == "file":
            source_path = Path(source.path)
            if not source_path.exists():
                self._warnings.append(f"Video file not found: {source.path}")

        if source.type == "rtsp" and not source.rtsp_url.startswith("rtsp://"):
            self._errors.append(f"Invalid RTSP URL: {source.rtsp_url}")

    def _validate_tracker(self) -> None:
        tracker = self._settings.tracker

        if tracker.type not in self.SUPPORTED_TRACKERS:
            self._errors.append(
                f"Unsupported tracker type '{tracker.type}'. Supported: {self.SUPPORTED_TRACKERS}"
            )

        if tracker.min_hits > tracker.max_age:
            self._errors.append("Tracker min_hits cannot exceed max_age")

    def _validate_output(self) -> None:
        output = self._settings.output

        if output.save_video:
            video_dir = Path(output.video_path).parent
            video_dir.mkdir(parents=True, exist_ok=True)

        if output.save_json:
            json_dir = Path(output.json_path).parent
            json_dir.mkdir(parents=True, exist_ok=True)

        if output.save_csv:
            csv_dir = Path(output.csv_path).parent
            csv_dir.mkdir(parents=True, exist_ok=True)

    def _validate_thresholds(self) -> None:
        model = self._settings.model

        if model.confidence_threshold < 0.1:
            self._warnings.append(
                "Very low confidence threshold may produce many false positives."
            )

        if model.confidence_threshold > 0.9:
            self._warnings.append(
                "Very high confidence threshold may miss valid detections."
            )

    def get_report(self) -> dict[str, Any]:
        return {
            "valid": self.is_valid,
            "errors": self._errors,
            "warnings": self._warnings,
            "settings_summary": {
                "device": self._settings.device,
                "model": self._settings.model.name,
                "source_type": self._settings.source.type,
                "tracker_type": self._settings.tracker.type,
                "visualization": self._settings.visualization.enabled,
            },
        }
