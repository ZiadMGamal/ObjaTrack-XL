from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class ModelSettings(BaseSettings):
    name: str = Field(default="yolov8n.pt")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=300, ge=1)
    classes: list[int] | None = Field(default=None)
    input_size: tuple[int, int] = Field(default=(640, 640))
    half_precision: bool = Field(default=False)
    warmup_iterations: int = Field(default=10, ge=0)

    @field_validator("input_size")
    @classmethod
    def validate_input_size(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] % 32 != 0 or v[1] % 32 != 0:
            raise ValueError("Input size must be divisible by 32")
        return v


class CaptureSettings(BaseSettings):
    type: str = Field(default="file")
    path: str = Field(default="data/samples/sample.mp4")
    webcam_index: int = Field(default=0, ge=0)
    rtsp_url: str = Field(default="rtsp://localhost:8554/stream")
    buffer_size: int = Field(default=128, ge=1)
    reconnect_attempts: int = Field(default=5, ge=0)
    reconnect_delay: float = Field(default=2.0, ge=0.0)
    width: int = Field(default=1280, ge=1)
    height: int = Field(default=720, ge=1)
    fps: float = Field(default=30.0, ge=1.0)
    loop: bool = Field(default=True)
    frame_skip: int = Field(default=0, ge=0)


class TrackerSettings(BaseSettings):
    type: str = Field(default="botsort")
    max_age: int = Field(default=30, ge=1)
    min_hits: int = Field(default=3, ge=1)
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_cosine_distance: float = Field(default=0.2, ge=0.0, le=1.0)
    nn_budget: int = Field(default=100, ge=1)
    track_buffer: int = Field(default=30, ge=1)
    track_high_thresh: float = Field(default=0.5, ge=0.0, le=1.0)
    track_low_thresh: float = Field(default=0.1, ge=0.0, le=1.0)
    new_track_thresh: float = Field(default=0.6, ge=0.0, le=1.0)
    match_thresh: float = Field(default=0.8, ge=0.0, le=1.0)
    with_reid: bool = Field(default=False)


class VisualizationSettings(BaseSettings):
    enabled: bool = Field(default=True)
    show_boxes: bool = Field(default=True)
    show_labels: bool = Field(default=True)
    show_confidence: bool = Field(default=True)
    show_tracks: bool = Field(default=True)
    show_trajectory: bool = Field(default=True)
    trajectory_length: int = Field(default=30, ge=1)
    show_hud: bool = Field(default=True)
    show_fps: bool = Field(default=True)
    box_thickness: int = Field(default=2, ge=1)
    font_scale: float = Field(default=0.6, ge=0.1)
    font_thickness: int = Field(default=1, ge=1)


class CountingSettings(BaseSettings):
    enabled: bool = Field(default=False)
    line_start: tuple[int, int] = Field(default=(0, 360))
    line_end: tuple[int, int] = Field(default=(1280, 360))


class ZoneSettings(BaseSettings):
    enabled: bool = Field(default=False)
    regions: list[list[tuple[int, int]]] = Field(default_factory=list)


class SpeedSettings(BaseSettings):
    enabled: bool = Field(default=False)
    pixels_per_meter: float = Field(default=8.0, ge=0.1)


class DwellSettings(BaseSettings):
    enabled: bool = Field(default=False)
    threshold_seconds: float = Field(default=10.0, ge=0.0)


class AnalyticsSettings(BaseSettings):
    counting: CountingSettings = Field(default_factory=CountingSettings)
    zones: ZoneSettings = Field(default_factory=ZoneSettings)
    speed_estimation: SpeedSettings = Field(default_factory=SpeedSettings)
    dwell_time: DwellSettings = Field(default_factory=DwellSettings)


class OutputSettings(BaseSettings):
    save_video: bool = Field(default=False)
    video_path: str = Field(default="outputs/videos/output.mp4")
    video_fps: int = Field(default=30, ge=1)
    video_codec: str = Field(default="mp4v")
    save_json: bool = Field(default=False)
    json_path: str = Field(default="outputs/exports/results.json")
    save_csv: bool = Field(default=False)
    csv_path: str = Field(default="outputs/exports/results.csv")


class LoggingSettings(BaseSettings):
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    file: str | None = Field(default="logs/objatrack.log")
    rotation: str = Field(default="10MB")
    retention: int = Field(default=7)

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid}")
        return v.upper()


class OptimizationExportSettings(BaseSettings):
    format: str = Field(default="onnx")
    opset_version: int = Field(default=17, ge=7)
    dynamic_axes: bool = Field(default=True)
    simplify: bool = Field(default=True)
    output_dir: str = Field(default="models/optimized")


class QuantizationSettings(BaseSettings):
    enabled: bool = Field(default=True)
    method: str = Field(default="dynamic")
    precision: str = Field(default="int8")
    calibration_samples: int = Field(default=100, ge=1)
    per_channel: bool = Field(default=True)

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid = {"dynamic", "static"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid quantization method: {v}")
        return v.lower()

    @field_validator("precision")
    @classmethod
    def validate_precision(cls, v: str) -> str:
        valid = {"int8", "fp16", "uint8"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid precision: {v}")
        return v.lower()


class APISettings(BaseSettings):
    enabled: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class BenchmarkRunSettings(BaseSettings):
    warmup_iterations: int = Field(default=50, ge=0)
    benchmark_iterations: int = Field(default=500, ge=1)
    batch_sizes: list[int] = Field(default_factory=lambda: [1])
    input_sizes: list[tuple[int, int]] = Field(default_factory=lambda: [(640, 640)])
    num_runs: int = Field(default=3, ge=1)


class Settings(BaseSettings):
    project_name: str = Field(default="ObjaTrack-XL")
    version: str = Field(default="1.0.0")
    author: str = Field(default="Ziad Mohamed Gamal")
    environment: str = Field(default="development")
    device: str = Field(default="auto")
    model: ModelSettings = Field(default_factory=ModelSettings)
    source: CaptureSettings = Field(default_factory=CaptureSettings)
    tracker: TrackerSettings = Field(default_factory=TrackerSettings)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)
    analytics: AnalyticsSettings = Field(default_factory=AnalyticsSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    optimization_export: OptimizationExportSettings = Field(default_factory=OptimizationExportSettings)
    quantization: QuantizationSettings = Field(default_factory=QuantizationSettings)
    api: APISettings = Field(default_factory=APISettings)
    benchmark: BenchmarkRunSettings = Field(default_factory=BenchmarkRunSettings)

    model_config = {"env_prefix": "OBJATRACK_", "env_nested_delimiter": "__"}

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        import yaml

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls._build_from_dict(data)

    @classmethod
    def _build_from_dict(cls, data: dict[str, Any]) -> Settings:
        flat: dict[str, Any] = {}

        if "project" in data:
            flat["project_name"] = data["project"].get("name", "ObjaTrack-XL")
            flat["version"] = data["project"].get("version", "1.0.0")
            flat["author"] = data["project"].get("author", "Ziad Mohamed Gamal")

        flat["environment"] = data.get("environment", "development")
        flat["device"] = data.get("device", "auto")

        if "model" in data:
            flat["model"] = ModelSettings(**data["model"])

        if "source" in data:
            flat["source"] = CaptureSettings(**data["source"])

        if "tracker" in data:
            flat["tracker"] = TrackerSettings(**data["tracker"])

        if "visualization" in data:
            flat["visualization"] = VisualizationSettings(**data["visualization"])

        if "analytics" in data:
            analytics_data = data["analytics"]
            analytics_kwargs: dict[str, Any] = {}
            if "counting" in analytics_data:
                analytics_kwargs["counting"] = CountingSettings(**analytics_data["counting"])
            if "zones" in analytics_data:
                analytics_kwargs["zones"] = ZoneSettings(**analytics_data["zones"])
            if "speed_estimation" in analytics_data:
                analytics_kwargs["speed_estimation"] = SpeedSettings(**analytics_data["speed_estimation"])
            if "dwell_time" in analytics_data:
                analytics_kwargs["dwell_time"] = DwellSettings(**analytics_data["dwell_time"])
            flat["analytics"] = AnalyticsSettings(**analytics_kwargs)

        if "output" in data:
            flat["output"] = OutputSettings(**data["output"])

        if "logging" in data:
            flat["logging"] = LoggingSettings(**data["logging"])

        if "api" in data:
            flat["api"] = APISettings(**data["api"])

        return cls(**flat)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
