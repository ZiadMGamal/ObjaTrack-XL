from __future__ import annotations

import numpy as np
import pytest

from src.core.base import DetectionResult
from src.tracking.track import Track, TrackState


@pytest.fixture
def sample_frame() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection() -> DetectionResult:
    return DetectionResult(
        boxes=np.array([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=np.float32),
        scores=np.array([0.9, 0.85], dtype=np.float32),
        class_ids=np.array([0, 1], dtype=np.int32),
        class_names=["person", "car"],
        frame_id=0,
        timestamp=1000.0,
    )


@pytest.fixture
def empty_detection() -> DetectionResult:
    return DetectionResult(
        boxes=np.empty((0, 4), dtype=np.float32),
        scores=np.empty((0,), dtype=np.float32),
        class_ids=np.empty((0,), dtype=np.int32),
        class_names=[],
        frame_id=0,
        timestamp=1000.0,
    )


@pytest.fixture
def sample_tracks() -> list[Track]:
    return [
        Track(
            track_id=1,
            box=np.array([100, 100, 200, 200], dtype=np.float32),
            score=0.9,
            class_id=0,
            class_name="person",
            state=TrackState.CONFIRMED,
            trajectory=[(150.0, 150.0)],
        ),
        Track(
            track_id=2,
            box=np.array([300, 300, 400, 400], dtype=np.float32),
            score=0.85,
            class_id=1,
            class_name="car",
            state=TrackState.CONFIRMED,
            trajectory=[(350.0, 350.0)],
        ),
    ]
