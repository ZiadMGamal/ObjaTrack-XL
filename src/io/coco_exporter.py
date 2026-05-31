from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class COCOExporter:

    def __init__(self, output_path: str = "outputs/exports/coco_results.json") -> None:
        self._output_path = Path(output_path)
        self._annotations: list[dict[str, Any]] = []
        self._images: list[dict[str, Any]] = []
        self._categories: dict[int, dict[str, Any]] = {}
        self._annotation_id: int = 1

    def add_category(self, category_id: int, name: str, supercategory: str = "") -> None:
        self._categories[category_id] = {
            "id": category_id,
            "name": name,
            "supercategory": supercategory or name,
        }

    def add_image(
        self,
        image_id: int,
        file_name: str,
        width: int,
        height: int,
    ) -> None:
        self._images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })

    def add_detection(
        self,
        image_id: int,
        category_id: int,
        bbox_xyxy: list[float],
        score: float,
        area: float | None = None,
    ) -> None:
        x1, y1, x2, y2 = bbox_xyxy
        w = x2 - x1
        h = y2 - y1
        bbox_xywh = [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)]

        self._annotations.append({
            "id": self._annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox_xywh,
            "area": round(area or (w * h), 1),
            "score": round(score, 4),
            "iscrowd": 0,
        })
        self._annotation_id += 1

    def add_frame_detections(
        self,
        frame_id: int,
        width: int,
        height: int,
        detections: list[dict[str, Any]],
    ) -> None:
        self.add_image(frame_id, f"frame_{frame_id:06d}.jpg", width, height)

        for det in detections:
            category_id = det.get("class_id", 0)
            if category_id not in self._categories:
                self.add_category(category_id, det.get("class_name", f"class_{category_id}"))

            self.add_detection(
                image_id=frame_id,
                category_id=category_id,
                bbox_xyxy=det.get("box", [0, 0, 0, 0]),
                score=det.get("score", 0),
            )

    def export(self) -> str:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        coco_format = {
            "images": self._images,
            "annotations": self._annotations,
            "categories": list(self._categories.values()),
        }

        with open(self._output_path, "w") as f:
            json.dump(coco_format, f, indent=2)

        logger.info(
            "coco_exported",
            path=str(self._output_path),
            images=len(self._images),
            annotations=len(self._annotations),
            categories=len(self._categories),
        )
        return str(self._output_path)

    def export_results_only(self) -> str:
        results_path = self._output_path.with_name(
            self._output_path.stem + "_results" + self._output_path.suffix
        )

        results = []
        for ann in self._annotations:
            results.append({
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "score": ann["score"],
            })

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        return str(results_path)

    def clear(self) -> None:
        self._annotations.clear()
        self._images.clear()
        self._categories.clear()
        self._annotation_id = 1
