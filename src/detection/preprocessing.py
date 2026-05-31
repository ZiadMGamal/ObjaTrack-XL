from __future__ import annotations

import cv2
import numpy as np

from src.utils.image_utils import ImageProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionPreprocessor:
    def __init__(
        self,
        input_size: tuple[int, int] = (640, 640),
        normalize: bool = True,
        mean: tuple[float, ...] = (0.0, 0.0, 0.0),
        std: tuple[float, ...] = (1.0, 1.0, 1.0),
        swap_rb: bool = True,
        letterbox: bool = True,
        pad_color: tuple[int, int, int] = (114, 114, 114),
        dtype: type = np.float32,
    ) -> None:
        self._input_size = input_size
        self._normalize = normalize
        self._mean = mean
        self._std = std
        self._swap_rb = swap_rb
        self._letterbox = letterbox
        self._pad_color = pad_color
        self._dtype = dtype

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        if self._swap_rb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self._letterbox:
            image, ratio, padding = ImageProcessor.letterbox(image, self._input_size, color=self._pad_color)
        else:
            ratio = min(
                self._input_size[0] / image.shape[0],
                self._input_size[1] / image.shape[1],
            )
            padding = (0.0, 0.0)
            image = cv2.resize(image, self._input_size[::-1], interpolation=cv2.INTER_LINEAR)

        if self._normalize:
            image = image.astype(self._dtype) / 255.0
            if any(m != 0.0 for m in self._mean) or any(s != 1.0 for s in self._std):
                mean_arr = np.array(self._mean, dtype=self._dtype).reshape(1, 1, 3)
                std_arr = np.array(self._std, dtype=self._dtype).reshape(1, 1, 3)
                image = (image - mean_arr) / std_arr

        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        image = np.expand_dims(image, axis=0).astype(self._dtype)

        return image, ratio, padding

    def preprocess_batch(self, images: list[np.ndarray]) -> tuple[np.ndarray, list[float], list[tuple[float, float]]]:
        batch = []
        ratios = []
        paddings = []

        for img in images:
            processed, ratio, padding = self.preprocess(img)
            batch.append(processed[0])
            ratios.append(ratio)
            paddings.append(padding)

        return np.stack(batch), ratios, paddings
