from __future__ import annotations


class ObjaTrackError(Exception):
    def __init__(self, message: str, error_code: str | None = None, details: dict | None = None) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        base = f"[{self.error_code}] {self.message}" if self.error_code else self.message
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{base} ({detail_str})"
        return base

    def to_dict(self) -> dict:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class CaptureError(ObjaTrackError):
    def __init__(self, message: str, source: str | None = None, **kwargs) -> None:
        details = kwargs.pop("details", {})
        if source:
            details["source"] = source
        super().__init__(message, error_code="CAPTURE_ERR", details=details, **kwargs)


class CaptureTimeoutError(CaptureError):
    def __init__(self, source: str, timeout: float) -> None:
        super().__init__(
            f"Capture timeout after {timeout}s",
            source=source,
            details={"timeout": timeout},
        )


class CaptureConnectionError(CaptureError):
    def __init__(self, source: str, reason: str | None = None) -> None:
        msg = f"Failed to connect to source: {source}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg, source=source)


class DetectionError(ObjaTrackError):
    def __init__(self, message: str, model_name: str | None = None, **kwargs) -> None:
        details = kwargs.pop("details", {})
        if model_name:
            details["model"] = model_name
        super().__init__(message, error_code="DETECT_ERR", details=details, **kwargs)


class DetectionPreprocessError(DetectionError):
    def __init__(self, message: str, input_shape: tuple | None = None) -> None:
        details = {}
        if input_shape:
            details["input_shape"] = str(input_shape)
        super().__init__(message, details=details)


class DetectionPostprocessError(DetectionError):
    def __init__(self, message: str, output_shape: tuple | None = None) -> None:
        details = {}
        if output_shape:
            details["output_shape"] = str(output_shape)
        super().__init__(message, details=details)


class TrackingError(ObjaTrackError):
    def __init__(self, message: str, tracker_type: str | None = None, **kwargs) -> None:
        details = kwargs.pop("details", {})
        if tracker_type:
            details["tracker"] = tracker_type
        super().__init__(message, error_code="TRACK_ERR", details=details, **kwargs)


class TrackingAssociationError(TrackingError):
    def __init__(self, message: str, num_detections: int = 0, num_tracks: int = 0) -> None:
        super().__init__(
            message,
            details={"num_detections": num_detections, "num_tracks": num_tracks},
        )


class OptimizationError(ObjaTrackError):
    def __init__(self, message: str, format_type: str | None = None, **kwargs) -> None:
        details = kwargs.pop("details", {})
        if format_type:
            details["format"] = format_type
        super().__init__(message, error_code="OPT_ERR", details=details, **kwargs)


class QuantizationError(OptimizationError):
    def __init__(self, message: str, precision: str | None = None) -> None:
        details = {}
        if precision:
            details["precision"] = precision
        super().__init__(message, format_type="quantization", details=details)


class ExportError(OptimizationError):
    def __init__(self, message: str, source_format: str = "", target_format: str = "") -> None:
        super().__init__(
            message,
            details={"source_format": source_format, "target_format": target_format},
        )


class ConfigurationError(ObjaTrackError):
    def __init__(self, message: str, config_key: str | None = None, **kwargs) -> None:
        details = kwargs.pop("details", {})
        if config_key:
            details["key"] = config_key
        super().__init__(message, error_code="CONFIG_ERR", details=details, **kwargs)


class ConfigValidationError(ConfigurationError):
    def __init__(self, field: str, value: object, reason: str) -> None:
        super().__init__(
            f"Validation failed for '{field}': {reason}",
            config_key=field,
            details={"value": str(value), "reason": reason},
        )


class ModelError(ObjaTrackError):
    def __init__(self, message: str, model_path: str | None = None, **kwargs) -> None:
        details = kwargs.pop("details", {})
        if model_path:
            details["model_path"] = model_path
        super().__init__(message, error_code="MODEL_ERR", details=details, **kwargs)


class ModelNotFoundError(ModelError):
    def __init__(self, model_path: str) -> None:
        super().__init__(f"Model not found: {model_path}", model_path=model_path)


class ModelLoadError(ModelError):
    def __init__(self, model_path: str, reason: str | None = None) -> None:
        msg = f"Failed to load model: {model_path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg, model_path=model_path)


class PipelineError(ObjaTrackError):
    def __init__(self, message: str, stage: str | None = None, **kwargs) -> None:
        details = kwargs.pop("details", {})
        if stage:
            details["stage"] = stage
        super().__init__(message, error_code="PIPELINE_ERR", details=details, **kwargs)


class PipelineStageError(PipelineError):
    def __init__(self, stage: str, original_error: Exception) -> None:
        super().__init__(
            f"Pipeline failed at stage '{stage}': {original_error}",
            stage=stage,
            details={"original_error": str(original_error), "error_type": type(original_error).__name__},
        )


class APIError(ObjaTrackError):
    def __init__(self, message: str, status_code: int = 500, **kwargs) -> None:
        details = kwargs.pop("details", {})
        details["status_code"] = status_code
        super().__init__(message, error_code="API_ERR", details=details, **kwargs)
        self.status_code = status_code


class ResourceExhaustedError(ObjaTrackError):
    def __init__(self, resource: str, limit: str | None = None) -> None:
        details = {"resource": resource}
        if limit:
            details["limit"] = limit
        super().__init__(
            f"Resource exhausted: {resource}",
            error_code="RESOURCE_ERR",
            details=details,
        )


class DeviceError(ObjaTrackError):
    def __init__(self, message: str, device: str | None = None) -> None:
        details = {}
        if device:
            details["device"] = device
        super().__init__(message, error_code="DEVICE_ERR", details=details)
