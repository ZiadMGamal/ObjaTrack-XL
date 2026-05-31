from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class JSONExporter:

    def __init__(self, output_path: str = "outputs/exports/results.json") -> None:
        self._output_path = Path(output_path)
        self._data: dict[str, Any] = {
            "metadata": {},
            "frames": [],
            "summary": {},
        }

    @property
    def output_path(self) -> Path:
        return self._output_path

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        self._data["metadata"] = metadata

    def add_frame_result(
        self,
        frame_id: int,
        timestamp: float,
        detections: list[dict[str, Any]],
        tracks: list[dict[str, Any]] | None = None,
        analytics: dict[str, Any] | None = None,
    ) -> None:
        frame_data: dict[str, Any] = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "detections": detections,
        }
        if tracks:
            frame_data["tracks"] = tracks
        if analytics:
            frame_data["analytics"] = analytics

        self._data["frames"].append(frame_data)

    def set_summary(self, summary: dict[str, Any]) -> None:
        self._data["summary"] = summary

    def export(self) -> str:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._output_path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

        logger.info(
            "json_exported",
            path=str(self._output_path),
            frames=len(self._data["frames"]),
        )
        return str(self._output_path)

    def export_streaming(self, frame_data: dict[str, Any]) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._output_path, "a") as f:
            f.write(json.dumps(frame_data, default=str) + "\n")

    def clear(self) -> None:
        self._data = {"metadata": {}, "frames": [], "summary": {}}
