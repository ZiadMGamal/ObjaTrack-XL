from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    score_threshold: float = 0.0,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    if score_threshold > 0:
        mask = scores > score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-7)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.3,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    method: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    boxes = boxes.copy().astype(np.float32)
    scores = scores.copy().astype(np.float32)
    n = len(boxes)
    indices = np.arange(n)

    for i in range(n):
        max_idx = i + np.argmax(scores[i:])

        boxes[[i, max_idx]] = boxes[[max_idx, i]]
        scores[[i, max_idx]] = scores[[max_idx, i]]
        indices[[i, max_idx]] = indices[[max_idx, i]]

        xx1 = np.maximum(boxes[i, 0], boxes[i + 1:, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[i + 1:, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[i + 1:, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[i + 1:, 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        areas_rest = (boxes[i + 1:, 2] - boxes[i + 1:, 0]) * (boxes[i + 1:, 3] - boxes[i + 1:, 1])
        union = area_i + areas_rest - intersection
        iou = intersection / np.maximum(union, 1e-7)

        if method == "gaussian":
            weight = np.exp(-(iou * iou) / sigma)
        elif method == "linear":
            weight = np.where(iou > iou_threshold, 1 - iou, np.ones_like(iou))
        else:
            weight = np.where(iou > iou_threshold, np.zeros_like(iou), np.ones_like(iou))

        scores[i + 1:] *= weight

    keep = np.where(scores > score_threshold)[0]
    return indices[keep], scores[keep]


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    unique_classes = np.unique(class_ids)
    keep_indices = []

    for cls_id in unique_classes:
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]

        cls_keep = non_max_suppression(cls_boxes, cls_scores, iou_threshold)
        keep_indices.extend(cls_indices[cls_keep].tolist())

    if not keep_indices:
        return np.array([], dtype=np.int64)

    keep_indices = np.array(keep_indices, dtype=np.int64)
    order = scores[keep_indices].argsort()[::-1]
    return keep_indices[order]


def diou_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    beta: float = 0.6,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-7)

        cxx1 = np.minimum(x1[i], x1[order[1:]])
        cyy1 = np.minimum(y1[i], y1[order[1:]])
        cxx2 = np.maximum(x2[i], x2[order[1:]])
        cyy2 = np.maximum(y2[i], y2[order[1:]])
        c_diag = (cxx2 - cxx1) ** 2 + (cyy2 - cyy1) ** 2

        d2 = (cx[i] - cx[order[1:]]) ** 2 + (cy[i] - cy[order[1:]]) ** 2
        diou = iou - (d2 / np.maximum(c_diag, 1e-7)) ** beta

        inds = np.where(diou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)
