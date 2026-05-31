from __future__ import annotations

import io
import time
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config.settings import Settings
from src.detection.detector_factory import DetectorFactory
from src.detection.base_detector import BaseObjectDetector
from src.tracking.tracker_factory import TrackerFactory
from src.tracking.base_tracker import BaseObjectTracker
from src.tracking.track import Track
from src.metrics.metrics_aggregator import MetricsAggregator
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="ObjaTrack-XL API",
    description="High-Performance Object Detection & Tracking REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector: BaseObjectDetector | None = None
_tracker: BaseObjectTracker | None = None
_metrics: MetricsAggregator | None = None
_initialized: bool = False


class DetectionRequest(BaseModel):
    confidence: float = 0.25
    iou_threshold: float = 0.45
    classes: list[int] | None = None


class DetectionResponse(BaseModel):
    num_detections: int
    detections: list[dict[str, Any]]
    latency_ms: float
    model: str


class TrackResponse(BaseModel):
    num_tracks: int
    tracks: list[dict[str, Any]]
    latency_ms: float


class SystemStatus(BaseModel):
    status: str
    initialized: bool
    model: str
    tracker: str
    uptime_seconds: float
    total_inferences: int


@app.on_event("startup")
async def startup_event() -> None:
    global _detector, _tracker, _metrics, _initialized

    settings = Settings()
    _detector = DetectorFactory.create_from_settings(settings.model)
    _detector.initialize()
    _detector.warmup(iterations=3)

    _tracker = TrackerFactory.create_from_settings(settings.tracker)
    _tracker.initialize()

    _metrics = MetricsAggregator()
    _metrics.fps_counter.start()
    _initialized = True

    logger.info("api_started", model=settings.model.name, tracker=settings.tracker.type)


@app.get("/", response_class=JSONResponse)
async def root() -> dict[str, str]:
    return {
        "name": "ObjaTrack-XL API",
        "version": "1.0.0",
        "author": "Ziad Mohamed Gamal",
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy" if _initialized else "initializing",
        "initialized": _initialized,
    }


@app.get("/status", response_model=SystemStatus)
async def status() -> SystemStatus:
    if not _initialized or _detector is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    summary = _metrics.get_summary() if _metrics else {}
    return SystemStatus(
        status="running",
        initialized=True,
        model=_detector._model_path,
        tracker=_tracker._name if _tracker else "none",
        uptime_seconds=summary.get("uptime_seconds", 0),
        total_inferences=_detector.inference_count,
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.0, le=1.0),
    iou_threshold: float = Query(0.45, ge=0.0, le=1.0),
) -> DetectionResponse:
    if not _initialized or _detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    start = time.perf_counter()
    result = _detector.detect(frame)
    latency = (time.perf_counter() - start) * 1000

    detections = []
    for i in range(result.num_detections):
        detections.append({
            "box": result.boxes[i].tolist(),
            "score": round(float(result.scores[i]), 4),
            "class_id": int(result.class_ids[i]),
            "class_name": result.class_names[i] if result.class_names else "",
        })

    if _metrics:
        _metrics.tick()

    return DetectionResponse(
        num_detections=result.num_detections,
        detections=detections,
        latency_ms=round(latency, 2),
        model=_detector._model_path,
    )


@app.post("/detect-and-track", response_model=TrackResponse)
async def detect_and_track(file: UploadFile = File(...)) -> TrackResponse:
    if not _initialized or _detector is None or _tracker is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    start = time.perf_counter()
    detections = _detector.detect(frame)
    tracks = _tracker.update(detections, frame)
    latency = (time.perf_counter() - start) * 1000

    track_list = [t.to_dict() for t in tracks]

    if _metrics:
        _metrics.tick()

    return TrackResponse(
        num_tracks=len(tracks),
        tracks=track_list,
        latency_ms=round(latency, 2),
    )


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    if _metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not available")
    return _metrics.get_snapshot()


@app.post("/reset-tracker")
async def reset_tracker() -> dict[str, str]:
    if _tracker is None:
        raise HTTPException(status_code=503, detail="Tracker not available")
    _tracker.reset()
    return {"status": "tracker_reset"}


@app.get("/info")
async def info() -> dict[str, Any]:
    from src.utils.device import DeviceManager
    dm = DeviceManager()
    return {
        "project": "ObjaTrack-XL",
        "version": "1.0.0",
        "author": "Ziad Mohamed Gamal",
        "system": dm.get_system_info(),
        "detectors": DetectorFactory.available_detectors(),
        "trackers": TrackerFactory.available_trackers(),
    }
