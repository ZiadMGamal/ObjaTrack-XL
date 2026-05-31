from __future__ import annotations

import math

import numpy as np


class MathUtils:

    @staticmethod
    def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def manhattan_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    @staticmethod
    def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
        y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
        x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
        y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

        union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - intersection
        union = np.maximum(union, 1e-7)

        return intersection / union

    @staticmethod
    def giou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        cx1 = min(box1[0], box2[0])
        cy1 = min(box1[1], box2[1])
        cx2 = max(box1[2], box2[2])
        cy2 = max(box1[3], box2[3])
        enclosing = (cx2 - cx1) * (cy2 - cy1)

        iou = intersection / union if union > 0 else 0.0
        giou_val = iou - (enclosing - union) / enclosing if enclosing > 0 else iou
        return float(giou_val)

    @staticmethod
    def diou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0.0

        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        d2 = (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2

        cx1 = min(box1[0], box2[0])
        cy1 = min(box1[1], box2[1])
        cx2 = max(box1[2], box2[2])
        cy2 = max(box1[3], box2[3])
        c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2

        return float(iou - d2 / c2) if c2 > 0 else float(iou)

    @staticmethod
    def box_area(box: np.ndarray) -> float:
        return float((box[2] - box[0]) * (box[3] - box[1]))

    @staticmethod
    def box_center(box: np.ndarray) -> tuple[float, float]:
        return (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))

    @staticmethod
    def box_aspect_ratio(box: np.ndarray) -> float:
        w = box[2] - box[0]
        h = box[3] - box[1]
        return float(w / h) if h > 0 else 0.0

    @staticmethod
    def line_intersection(
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        p4: tuple[float, float],
    ) -> tuple[float, float] | None:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (ix, iy)

        return None

    @staticmethod
    def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def angle_between_points(
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float:
        return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

    @staticmethod
    def moving_average(values: list[float], window: int) -> list[float]:
        if len(values) < window:
            return values
        result = []
        for i in range(len(values) - window + 1):
            avg = sum(values[i : i + window]) / window
            result.append(avg)
        return result

    @staticmethod
    def exponential_moving_average(values: list[float], alpha: float = 0.3) -> list[float]:
        if not values:
            return []
        result = [values[0]]
        for v in values[1:]:
            result.append(alpha * v + (1 - alpha) * result[-1])
        return result

    @staticmethod
    def gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
        x = np.arange(size) - (size - 1) / 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel_2d = np.outer(kernel, kernel)
        return kernel_2d / kernel_2d.sum()
