from __future__ import annotations

import numpy as np

from src.detection.preprocessing import DetectionPreprocessor


class TestDetectionPreprocessor:
    def test_initialization(self) -> None:
        prep = DetectionPreprocessor(input_size=(640, 640))
        assert prep.input_size == (640, 640)

    def test_preprocess_shape(self) -> None:
        prep = DetectionPreprocessor(input_size=(640, 640))
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, ratio, padding = prep.preprocess(image)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32

    def test_preprocess_normalization(self) -> None:
        prep = DetectionPreprocessor(input_size=(640, 640), normalize=True)
        image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        tensor, _, _ = prep.preprocess(image)
        assert tensor.max() <= 1.0

    def test_preprocess_no_normalization(self) -> None:
        prep = DetectionPreprocessor(input_size=(640, 640), normalize=False)
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        tensor, _, _ = prep.preprocess(image)
        assert tensor.max() > 1.0

    def test_preprocess_batch(self) -> None:
        prep = DetectionPreprocessor(input_size=(320, 320))
        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
        ]
        batch, ratios, paddings = prep.preprocess_batch(images)
        assert batch.shape == (2, 3, 320, 320)
        assert len(ratios) == 2
        assert len(paddings) == 2

    def test_preprocess_square_image(self) -> None:
        prep = DetectionPreprocessor(input_size=(640, 640))
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        tensor, ratio, padding = prep.preprocess(image)
        assert tensor.shape == (1, 3, 640, 640)

    def test_no_letterbox(self) -> None:
        prep = DetectionPreprocessor(input_size=(640, 640), letterbox=False)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, ratio, padding = prep.preprocess(image)
        assert tensor.shape == (1, 3, 640, 640)
