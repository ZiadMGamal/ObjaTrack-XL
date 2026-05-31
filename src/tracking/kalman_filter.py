from __future__ import annotations

import numpy as np


class KalmanFilterXYAH:

    def __init__(self) -> None:
        ndim = 4
        dt = 1.0

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov

        return projected_mean, projected_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.solve(
            chol_factor,
            np.linalg.solve(chol_factor, (covariance @ self._update_mat.T).T).T,
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T

        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        projected_mean, projected_cov = self.project(mean, covariance)

        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]

        diff = measurements - projected_mean

        try:
            cholesky = np.linalg.cholesky(projected_cov)
            z = np.linalg.solve(cholesky, diff.T).T
            return np.sum(z * z, axis=1)
        except np.linalg.LinAlgError:
            return np.full(len(measurements), 1e18)


class KalmanFilterXYWH:

    def __init__(self) -> None:
        self._dt = 1.0
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cx, cy, w, h = measurement
        mean = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(8)
        for i in range(4):
            F[i, i + 4] = self._dt

        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
        ]
        Q = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = F @ mean
        covariance = F @ covariance @ F.T + Q
        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        H = np.eye(4, 8)
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
        ]
        R = np.diag(np.square(std))

        projected_mean = H @ mean
        projected_cov = H @ covariance @ H.T + R

        K = covariance @ H.T @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean

        new_mean = mean + K @ innovation
        new_covariance = (np.eye(8) - K @ H) @ covariance
        return new_mean, new_covariance
