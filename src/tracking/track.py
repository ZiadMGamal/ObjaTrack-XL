from __future__ import annotations

from enum import IntEnum
from typing import Any
from dataclasses import dataclass, field

import numpy as np


class TrackState(IntEnum):
    TENTATIVE = 0
    CONFIRMED = 1
    COASTED = 2
    DELETED = 3


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    score: float
    class_id: int
    class_name: str = ""
    state: TrackState = TrackState.TENTATIVE
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    features: list[np.ndarray] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _start_frame: int = 0
    _last_frame: int = 0

    @property
    def center(self) -> tuple[float, float]:
        return (
            float((self.box[0] + self.box[2]) / 2),
            float((self.box[1] + self.box[3]) / 2),
        )

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED

    @property
    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    @property
    def track_length(self) -> int:
        return len(self.trajectory)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def direction(self) -> float:
        if self.speed < 1e-6:
            return 0.0
        return float(np.degrees(np.arctan2(self.velocity[1], self.velocity[0])))

    def update_box(self, box: np.ndarray, score: float, class_id: int, class_name: str = "") -> None:
        old_center = self.center
        self.box = box
        self.score = score
        self.class_id = class_id
        if class_name:
            self.class_name = class_name
        self.hits += 1
        self.time_since_update = 0
        self.age += 1

        new_center = self.center
        self.velocity = np.array([
            new_center[0] - old_center[0],
            new_center[1] - old_center[1],
        ])
        self.trajectory.append(new_center)

    def mark_missed(self) -> None:
        self.time_since_update += 1
        self.age += 1

    def predict_next_position(self) -> np.ndarray:
        predicted = self.box.copy()
        predicted[0] += self.velocity[0]
        predicted[1] += self.velocity[1]
        predicted[2] += self.velocity[0]
        predicted[3] += self.velocity[1]
        return predicted

    def add_feature(self, feature: np.ndarray, max_features: int = 100) -> None:
        self.features.append(feature)
        if len(self.features) > max_features:
            self.features = self.features[-max_features:]

    def get_latest_feature(self) -> np.ndarray | None:
        return self.features[-1] if self.features else None

    def get_trajectory_array(self) -> np.ndarray:
        if not self.trajectory:
            return np.empty((0, 2))
        return np.array(self.trajectory)

    def to_tlbr(self) -> np.ndarray:
        return self.box.copy()

    def to_tlwh(self) -> np.ndarray:
        ret = self.box.copy()
        ret[2] -= ret[0]
        ret[3] -= ret[1]
        return ret

    def to_xyah(self) -> np.ndarray:
        cx = (self.box[0] + self.box[2]) / 2
        cy = (self.box[1] + self.box[3]) / 2
        w = self.box[2] - self.box[0]
        h = self.box[3] - self.box[1]
        return np.array([cx, cy, w / h if h > 0 else 0, h])

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "box": self.box.tolist(),
            "score": round(self.score, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "state": self.state.name,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "center": self.center,
            "velocity": self.velocity.tolist(),
            "speed": round(self.speed, 2),
            "direction": round(self.direction, 2),
            "trajectory_length": self.track_length,
        }

    def __repr__(self) -> str:
        return (
            f"Track(id={self.track_id}, class={self.class_name}, "
            f"state={self.state.name}, hits={self.hits}, age={self.age})"
        )
