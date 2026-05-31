from __future__ import annotations

import numpy as np

from src.tracking.kalman_filter import KalmanFilterXYAH, KalmanFilterXYWH


class TestKalmanFilterXYAH:
    def test_initiate(self) -> None:
        kf = KalmanFilterXYAH()
        measurement = np.array([100.0, 100.0, 1.5, 50.0])
        mean, cov = kf.initiate(measurement)
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)
        assert mean[0] == 100.0
        assert mean[1] == 100.0
        assert mean[4] == 0.0

    def test_predict(self) -> None:
        kf = KalmanFilterXYAH()
        measurement = np.array([100.0, 100.0, 1.5, 50.0])
        mean, cov = kf.initiate(measurement)
        mean_pred, cov_pred = kf.predict(mean, cov)
        assert mean_pred.shape == (8,)
        assert cov_pred.shape == (8, 8)

    def test_update(self) -> None:
        kf = KalmanFilterXYAH()
        measurement = np.array([100.0, 100.0, 1.5, 50.0])
        mean, cov = kf.initiate(measurement)
        mean, cov = kf.predict(mean, cov)
        new_measurement = np.array([102.0, 101.0, 1.5, 50.0])
        mean_upd, cov_upd = kf.update(mean, cov, new_measurement)
        assert mean_upd.shape == (8,)
        assert abs(mean_upd[0] - 102.0) < 5.0

    def test_gating_distance(self) -> None:
        kf = KalmanFilterXYAH()
        measurement = np.array([100.0, 100.0, 1.5, 50.0])
        mean, cov = kf.initiate(measurement)

        measurements = np.array(
            [
                [100.0, 100.0, 1.5, 50.0],
                [200.0, 200.0, 2.0, 60.0],
            ]
        )
        distances = kf.gating_distance(mean, cov, measurements)
        assert distances.shape == (2,)
        assert distances[0] < distances[1]

    def test_predict_update_cycle(self) -> None:
        kf = KalmanFilterXYAH()
        measurement = np.array([100.0, 100.0, 1.5, 50.0])
        mean, cov = kf.initiate(measurement)

        for i in range(10):
            mean, cov = kf.predict(mean, cov)
            new_meas = np.array([100.0 + i, 100.0 + i * 0.5, 1.5, 50.0])
            mean, cov = kf.update(mean, cov, new_meas)

        assert mean.shape == (8,)


class TestKalmanFilterXYWH:
    def test_initiate(self) -> None:
        kf = KalmanFilterXYWH()
        measurement = np.array([100.0, 100.0, 80.0, 120.0])
        mean, cov = kf.initiate(measurement)
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)
        assert mean[0] == 100.0
        assert mean[2] == 80.0

    def test_predict(self) -> None:
        kf = KalmanFilterXYWH()
        measurement = np.array([100.0, 100.0, 80.0, 120.0])
        mean, cov = kf.initiate(measurement)
        mean_pred, cov_pred = kf.predict(mean, cov)
        assert mean_pred.shape == (8,)

    def test_update(self) -> None:
        kf = KalmanFilterXYWH()
        measurement = np.array([100.0, 100.0, 80.0, 120.0])
        mean, cov = kf.initiate(measurement)
        mean, cov = kf.predict(mean, cov)
        new_meas = np.array([105.0, 102.0, 82.0, 122.0])
        mean_upd, cov_upd = kf.update(mean, cov, new_meas)
        assert mean_upd.shape == (8,)
