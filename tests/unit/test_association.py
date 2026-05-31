from __future__ import annotations

import numpy as np
import pytest

from src.tracking.association import (
    iou_distance,
    euclidean_distance,
    linear_assignment,
    greedy_assignment,
    fuse_iou_score,
)


class TestIoUDistance:

    def test_empty_inputs(self) -> None:
        result = iou_distance(np.empty((0, 4)), np.array([[10, 10, 50, 50]]))
        assert result.shape[0] == 0

    def test_identical_boxes(self) -> None:
        boxes = np.array([[10, 10, 50, 50]])
        result = iou_distance(boxes, boxes)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(0.0, abs=0.01)

    def test_no_overlap(self) -> None:
        a = np.array([[0, 0, 50, 50]])
        b = np.array([[100, 100, 150, 150]])
        result = iou_distance(a, b)
        assert result[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_matrix_shape(self) -> None:
        a = np.array([[0, 0, 50, 50], [60, 60, 100, 100]])
        b = np.array([[10, 10, 40, 40], [70, 70, 110, 110], [200, 200, 250, 250]])
        result = iou_distance(a, b)
        assert result.shape == (2, 3)


class TestEuclideanDistance:

    def test_empty_inputs(self) -> None:
        result = euclidean_distance(np.empty((0, 2)), np.array([[1, 1]]))
        assert result.shape[0] == 0

    def test_same_point(self) -> None:
        points = np.array([[5.0, 5.0]])
        result = euclidean_distance(points, points)
        assert result[0, 0] == pytest.approx(0.0)

    def test_known_distance(self) -> None:
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        result = euclidean_distance(a, b)
        assert result[0, 0] == pytest.approx(5.0)


class TestLinearAssignment:

    def test_empty_matrix(self) -> None:
        cost = np.empty((0, 0))
        matches, ut, ud = linear_assignment(cost)
        assert len(matches) == 0

    def test_perfect_assignment(self) -> None:
        cost = np.array([
            [0.1, 0.9],
            [0.9, 0.1],
        ])
        matches, ut, ud = linear_assignment(cost, threshold=0.5)
        assert len(matches) == 2
        assert len(ut) == 0
        assert len(ud) == 0

    def test_threshold_filtering(self) -> None:
        cost = np.array([
            [0.9, 0.9],
            [0.9, 0.9],
        ])
        matches, ut, ud = linear_assignment(cost, threshold=0.5)
        assert len(matches) == 0
        assert len(ut) == 2
        assert len(ud) == 2

    def test_uneven_dimensions(self) -> None:
        cost = np.array([
            [0.1, 0.5, 0.9],
            [0.9, 0.1, 0.5],
        ])
        matches, ut, ud = linear_assignment(cost, threshold=0.5)
        assert len(matches) == 2
        assert len(ud) == 1


class TestGreedyAssignment:

    def test_empty_matrix(self) -> None:
        cost = np.empty((0, 0))
        matches, ut, ud = greedy_assignment(cost)
        assert len(matches) == 0

    def test_simple_assignment(self) -> None:
        cost = np.array([
            [0.1, 0.9],
            [0.9, 0.1],
        ])
        matches, ut, ud = greedy_assignment(cost, threshold=0.5)
        assert len(matches) == 2


class TestFuseIoUScore:

    def test_empty(self) -> None:
        cost = np.empty((0, 0))
        scores = np.array([])
        result = fuse_iou_score(cost, scores)
        assert result.shape == (0, 0)

    def test_fuse(self) -> None:
        cost = np.array([[0.5, 0.3]])
        scores = np.array([0.9, 0.5])
        result = fuse_iou_score(cost, scores, alpha=0.5)
        assert result.shape == (1, 2)
