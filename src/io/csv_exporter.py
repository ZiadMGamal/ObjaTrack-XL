from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CSVExporter:

    DETECTION_HEADERS = [
        "frame_id", "timestamp", "detection_id",
        "x1", "y1", "x2", "y2",
        "confidence", "class_id", "class_name",
    ]

    TRACK_HEADERS = [
        "frame_id", "timestamp", "track_id",
        "x1", "y1", "x2", "y2",
        "confidence", "class_id", "class_name",
        "age", "hits", "speed", "direction",
    ]

    def __init__(
        self,
        output_path: str = "outputs/exports/results.csv",
        mode: str = "tracks",
    ) -> None:
        self._output_path = Path(output_path)
        self._mode = mode
        self._rows: list[list[Any]] = []
        self._headers = self.TRACK_HEADERS if mode == "tracks" else self.DETECTION_HEADERS

    @property
    def output_path(self) -> Path:
        return self._output_path

    def add_detections(
        self,
        frame_id: int,
        timestamp: float,
        detections: list[dict[str, Any]],
    ) -> None:
        for i, det in enumerate(detections):
            row = [
                frame_id, round(timestamp, 6), i,
                det.get("box", [0, 0, 0, 0])[0],
                det.get("box", [0, 0, 0, 0])[1],
                det.get("box", [0, 0, 0, 0])[2],
                det.get("box", [0, 0, 0, 0])[3],
                round(det.get("score", 0), 4),
                det.get("class_id", 0),
                det.get("class_name", ""),
            ]
            self._rows.append(row)

    def add_tracks(
        self,
        frame_id: int,
        timestamp: float,
        tracks: list[dict[str, Any]],
    ) -> None:
        for track in tracks:
            box = track.get("box", [0, 0, 0, 0])
            row = [
                frame_id, round(timestamp, 6), track.get("track_id", 0),
                round(box[0], 1), round(box[1], 1),
                round(box[2], 1), round(box[3], 1),
                round(track.get("score", 0), 4),
                track.get("class_id", 0),
                track.get("class_name", ""),
                track.get("age", 0),
                track.get("hits", 0),
                round(track.get("speed", 0), 2),
                round(track.get("direction", 0), 1),
            ]
            self._rows.append(row)

    def export(self) -> str:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._headers)
            writer.writerows(self._rows)

        logger.info(
            "csv_exported",
            path=str(self._output_path),
            rows=len(self._rows),
        )
        return str(self._output_path)

    def clear(self) -> None:
        self._rows.clear()
