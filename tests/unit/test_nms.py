from __future__ import annotations

import numpy as np
import pytest

from src.detection.nms import non_max_suppression, soft_nms, batched_nms, diou_nms


class TestNonMaxSuppression:

    def test_empty_input(self) -> None:
        boxes = np.array([])
        scores = np.array([])
        result = non_max_suppression(boxes, scores)
        assert len(result) == 0

    def test_single_box(self) -> None:
        boxes = np.array([[10, 10, 50, 50]])
        scores = np.array([0.9])
        result = non_max_suppression(boxes, scores)
        assert len(result) == 1
        assert result[0] == 0

    def test_no_overlap(self) -> None:
        boxes = np.array([
            [10, 10, 50, 50],
            [100, 100, 150, 150],
            [200, 200, 250, 250],
        ])
        scores = np.array([0.9, 0.8, 0.7])
        result = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert len(result) == 3

    def test_high_overlap_suppression(self) -> None:
        boxes = np.array([
            [10, 10, 50, 50],
            [11, 11, 51, 51],
            [12, 12, 52, 52],
        ])
        scores = np.array([0.9, 0.8, 0.7])
        result = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert len(result) == 1
        assert result[0] == 0

    def test_partial_overlap(self) -> None:
        boxes = np.array([
            [0, 0, 100, 100],
            [50, 50, 150, 150],
        ])
        scores = np.array([0.9, 0.85])
        result = non_max_suppression(boxes, scores, iou_threshold=0.3)
        assert len(result) >= 1

    def test_score_ordering(self) -> None:
        boxes = np.array([
            [10, 10, 50, 50],
            [100, 100, 150, 150],
        ])
        scores = np.array([0.5, 0.9])
        result = non_max_suppression(boxes, scores)
        assert result[0] == 1

    def test_score_threshold(self) -> None:
        boxes = np.array([
            [10, 10, 50, 50],
            [100, 100, 150, 150],
        ])
        scores = np.array([0.9, 0.01])
        result = non_max_suppression(boxes, scores, score_threshold=0.1)
        assert len(result) == 1


class TestSoftNMS:

    def test_empty_input(self) -> None:
        boxes = np.array([])
        scores = np.array([])
        keep, new_scores = soft_nms(boxes, scores)
        assert len(keep) == 0

    def test_single_box(self) -> None:
        boxes = np.array([[10, 10, 50, 50]])
        scores = np.array([0.9])
        keep, new_scores = soft_nms(boxes, scores)
        assert len(keep) == 1

    def test_gaussian_method(self) -> None:
        boxes = np.array([
            [10, 10, 50, 50],
            [12, 12, 52, 52],
            [200, 200, 250, 250],
        ])
        scores = np.array([0.9, 0.8, 0.7])
        keep, new_scores = soft_nms(boxes, scores, method="gaussian")
        assert len(keep) >= 2

    def test_linear_method(self) -> None:
        boxes = np.array([
            [10, 10, 50, 50],
            [12, 12, 52, 52],
        ])
        scores = np.array([0.9, 0.8])
        keep, new_scores = soft_nms(boxes, scores, method="linear")
        assert len(keep) >= 1


class TestBatchedNMS:

    def test_empty_input(self) -> None:
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        class_ids = np.array([])
        result = batched_nms(boxes, scores, class_ids)
        assert len(result) == 0

    def test_different_classes(self) -> None:
        boxes = np.array([
            [10, 10, 50, 50],
            [12, 12, 52, 52],
        ])
        scores = np.array([0.9, 0.8])
        class_ids = np.array([0, 1])
        result = batched_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert len(result) == 2

    def test_same_class_overlap(self) -> None:
        boxes = np.array([
            [10, 10, 50, 50],
            [12, 12, 52, 52],
        ])
        scores = np.array([0.9, 0.8])
        class_ids = np.array([0, 0])
        result = batched_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert len(result) == 1


class TestDIoUNMS:

    def test_empty_input(self) -> None:
        result = diou_nms(np.array([]).reshape(0, 4), np.array([]))
        assert len(result) == 0

    def test_single_box(self) -> None:
        boxes = np.array([[10, 10, 50, 50]])
        scores = np.array([0.9])
        result = diou_nms(boxes, scores)
        assert len(result) == 1

    def test_distant_boxes(self) -> None:
        boxes = np.array([
            [10, 10, 50, 50],
            [200, 200, 250, 250],
        ])
        scores = np.array([0.9, 0.8])
        result = diou_nms(boxes, scores)
        assert len(result) == 2
