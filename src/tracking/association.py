from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.utils.logger import get_logger
from src.utils.math_utils import MathUtils

logger = get_logger(__name__)


def iou_distance(tracks_boxes: np.ndarray, detection_boxes: np.ndarray) -> np.ndarray:
    if len(tracks_boxes) == 0 or len(detection_boxes) == 0:
        return np.empty((len(tracks_boxes), len(detection_boxes)))

    iou_matrix = MathUtils.iou_matrix(tracks_boxes, detection_boxes)
    return 1.0 - iou_matrix


def cosine_distance(track_features: list[np.ndarray], detection_features: np.ndarray) -> np.ndarray:
    if not track_features or len(detection_features) == 0:
        return np.zeros((len(track_features), len(detection_features)))

    cost_matrix = np.zeros((len(track_features), len(detection_features)))
    for i, track_feat in enumerate(track_features):
        if track_feat is None or len(track_feat) == 0:
            cost_matrix[i, :] = 1.0
            continue
        for j, det_feat in enumerate(detection_features):
            similarity = MathUtils.cosine_similarity(track_feat, det_feat)
            cost_matrix[i, j] = 1.0 - similarity

    return cost_matrix


def euclidean_distance(tracks_centers: np.ndarray, detection_centers: np.ndarray) -> np.ndarray:
    if len(tracks_centers) == 0 or len(detection_centers) == 0:
        return np.empty((len(tracks_centers), len(detection_centers)))

    diff = tracks_centers[:, np.newaxis, :] - detection_centers[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def mahalanobis_distance(
    tracks_means: np.ndarray,
    tracks_covariances: np.ndarray,
    measurements: np.ndarray,
) -> np.ndarray:
    n_tracks = len(tracks_means)
    n_measurements = len(measurements)
    cost_matrix = np.zeros((n_tracks, n_measurements))

    for i in range(n_tracks):
        diff = measurements - tracks_means[i]
        try:
            cov_inv = np.linalg.inv(tracks_covariances[i])
            for j in range(n_measurements):
                d = diff[j]
                cost_matrix[i, j] = np.sqrt(d @ cov_inv @ d)
        except np.linalg.LinAlgError:
            cost_matrix[i, :] = 1e5

    return cost_matrix


def linear_assignment(
    cost_matrix: np.ndarray, threshold: float = 0.7
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = list(range(cost_matrix.shape[0]))
    unmatched_detections = list(range(cost_matrix.shape[1]))

    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] > threshold:
            continue
        matches.append((row, col))
        if row in unmatched_tracks:
            unmatched_tracks.remove(row)
        if col in unmatched_detections:
            unmatched_detections.remove(col)

    return matches, unmatched_tracks, unmatched_detections


def greedy_assignment(
    cost_matrix: np.ndarray, threshold: float = 0.7
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    matches = []
    matched_rows = set()
    matched_cols = set()

    flat_indices = np.argsort(cost_matrix, axis=None)
    rows, cols = np.unravel_index(flat_indices, cost_matrix.shape)

    for row, col in zip(rows, cols):
        if cost_matrix[row, col] > threshold:
            break
        if row in matched_rows or col in matched_cols:
            continue
        matches.append((int(row), int(col)))
        matched_rows.add(row)
        matched_cols.add(col)

    unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
    unmatched_detections = [i for i in range(cost_matrix.shape[1]) if i not in matched_cols]

    return matches, unmatched_tracks, unmatched_detections


def fuse_iou_score(cost_matrix: np.ndarray, scores: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if cost_matrix.size == 0:
        return cost_matrix
    score_matrix = scores[np.newaxis, :].repeat(cost_matrix.shape[0], axis=0)
    fused = cost_matrix * (1 - alpha) + (1 - score_matrix) * alpha
    return fused


def gate_cost_matrix(
    cost_matrix: np.ndarray,
    tracks_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    gate_threshold: float = 0.5,
) -> np.ndarray:
    if cost_matrix.size == 0:
        return cost_matrix

    iou_dist = iou_distance(tracks_boxes, detection_boxes)
    gate_mask = iou_dist > gate_threshold
    cost_matrix[gate_mask] = 1e5

    return cost_matrix
