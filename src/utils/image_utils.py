from __future__ import annotations

import cv2
import numpy as np


class ImageProcessor:

    @staticmethod
    def letterbox(
        image: np.ndarray,
        target_size: tuple[int, int] = (640, 640),
        color: tuple[int, int, int] = (114, 114, 114),
        auto: bool = False,
        scale_fill: bool = False,
        stride: int = 32,
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        shape = image.shape[:2]
        target_h, target_w = target_size

        r = min(target_h / shape[0], target_w / shape[1])
        r = min(r, 1.0)

        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = target_w - new_unpad[0]
        dh = target_h - new_unpad[1]

        if auto:
            dw = dw % stride
            dh = dh % stride
        elif scale_fill:
            dw = 0.0
            dh = 0.0
            new_unpad = (target_w, target_h)
            r = min(target_w / shape[1], target_h / shape[0])

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color,
        )

        return image, r, (dw, dh)

    @staticmethod
    def normalize(image: np.ndarray, mean: tuple[float, ...] = (0.0, 0.0, 0.0), std: tuple[float, ...] = (1.0, 1.0, 1.0)) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        if any(m != 0.0 for m in mean) or any(s != 1.0 for s in std):
            mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
            std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
            img = (img - mean_arr) / std_arr
        return img

    @staticmethod
    def to_tensor(image: np.ndarray) -> np.ndarray:
        img = image.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img

    @staticmethod
    def prepare_input(
        image: np.ndarray,
        target_size: tuple[int, int] = (640, 640),
        normalize: bool = True,
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        img, ratio, padding = ImageProcessor.letterbox(image, target_size)
        if normalize:
            img = ImageProcessor.normalize(img)
        img = ImageProcessor.to_tensor(img)
        img = np.expand_dims(img, axis=0)
        return img, ratio, padding

    @staticmethod
    def scale_boxes(
        boxes: np.ndarray,
        original_shape: tuple[int, int],
        target_shape: tuple[int, int],
        ratio: float | None = None,
        padding: tuple[float, float] | None = None,
    ) -> np.ndarray:
        if ratio is not None and padding is not None:
            boxes = boxes.copy()
            boxes[:, [0, 2]] -= padding[0]
            boxes[:, [1, 3]] -= padding[1]
            boxes[:, :4] /= ratio
        else:
            gain = min(target_shape[0] / original_shape[0], target_shape[1] / original_shape[1])
            pad_x = (target_shape[1] - original_shape[1] * gain) / 2
            pad_y = (target_shape[0] - original_shape[0] * gain) / 2
            boxes = boxes.copy()
            boxes[:, [0, 2]] -= pad_x
            boxes[:, [1, 3]] -= pad_y
            boxes[:, :4] /= gain

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_shape[0])

        return boxes

    @staticmethod
    def crop_region(image: np.ndarray, box: np.ndarray, padding: float = 0.0) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box[:4].astype(int)

        if padding > 0:
            pw = int((x2 - x1) * padding)
            ph = int((y2 - y1) * padding)
            x1 = max(0, x1 - pw)
            y1 = max(0, y1 - ph)
            x2 = min(w, x2 + pw)
            y2 = min(h, y2 + ph)

        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def resize_aspect_ratio(image: np.ndarray, max_size: int) -> np.ndarray:
        h, w = image.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return image

    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
        result = boxes.copy()
        result[:, 2] = boxes[:, 2] - boxes[:, 0]
        result[:, 3] = boxes[:, 3] - boxes[:, 1]
        return result

    @staticmethod
    def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        result = boxes.copy()
        result[:, 2] = boxes[:, 0] + boxes[:, 2]
        result[:, 3] = boxes[:, 1] + boxes[:, 3]
        return result

    @staticmethod
    def xcycwh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        result = np.zeros_like(boxes)
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return result

    @staticmethod
    def xyxy_to_xcycwh(boxes: np.ndarray) -> np.ndarray:
        result = np.zeros_like(boxes)
        result[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        result[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        result[:, 2] = boxes[:, 2] - boxes[:, 0]
        result[:, 3] = boxes[:, 3] - boxes[:, 1]
        return result
