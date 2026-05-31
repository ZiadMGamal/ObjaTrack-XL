from __future__ import annotations

import numpy as np
import pytest

from src.utils.math_utils import MathUtils
from src.utils.image_utils import ImageProcessor


class TestMathUtils:

    def test_iou_identical(self) -> None:
        box = np.array([10, 10, 50, 50])
        assert MathUtils.compute_iou(box, box) == pytest.approx(1.0)

    def test_iou_no_overlap(self) -> None:
        a = np.array([0, 0, 10, 10])
        b = np.array([20, 20, 30, 30])
        assert MathUtils.compute_iou(a, b) == pytest.approx(0.0)

    def test_iou_partial(self) -> None:
        a = np.array([0, 0, 10, 10])
        b = np.array([5, 5, 15, 15])
        iou = MathUtils.compute_iou(a, b)
        assert 0 < iou < 1

    def test_iou_matrix(self) -> None:
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
        b = np.array([[5, 5, 15, 15], [25, 25, 35, 35]])
        matrix = MathUtils.iou_matrix(a, b)
        assert matrix.shape == (2, 2)

    def test_cosine_similarity(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert MathUtils.cosine_similarity(a, b) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert MathUtils.cosine_similarity(a, b) == pytest.approx(0.0, abs=0.01)

    def test_point_in_polygon(self) -> None:
        polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        assert MathUtils.point_in_polygon((5.0, 5.0), polygon) is True
        assert MathUtils.point_in_polygon((15.0, 5.0), polygon) is False

    def test_line_intersection(self) -> None:
        result = MathUtils.line_intersection(
            (0.0, 0.0), (10.0, 10.0),
            (0.0, 10.0), (10.0, 0.0),
        )
        assert result is not None
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(5.0)

    def test_no_intersection(self) -> None:
        result = MathUtils.line_intersection(
            (0.0, 0.0), (1.0, 0.0),
            (0.0, 1.0), (1.0, 1.0),
        )
        assert result is None


class TestImageProcessor:

    def test_letterbox_shape(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result, ratio, padding = ImageProcessor.letterbox(image, (640, 640))
        assert result.shape[:2] == (640, 640)

    def test_letterbox_square(self) -> None:
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result, ratio, padding = ImageProcessor.letterbox(image, (640, 640))
        assert result.shape[:2] == (640, 640)

    def test_normalize_image(self) -> None:
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = ImageProcessor.normalize(image)
        assert result.dtype == np.float32
        assert result.max() <= 1.0

    def test_xyxy_to_xywh(self) -> None:
        xyxy = np.array([10, 20, 110, 120])
        xywh = ImageProcessor.xyxy_to_xywh(xyxy)
        assert xywh[0] == 60
        assert xywh[1] == 70
        assert xywh[2] == 100
        assert xywh[3] == 100

    def test_xywh_to_xyxy(self) -> None:
        xywh = np.array([60, 70, 100, 100])
        xyxy = ImageProcessor.xywh_to_xyxy(xywh)
        assert xyxy[0] == 10
        assert xyxy[1] == 20
