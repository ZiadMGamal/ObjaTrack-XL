from __future__ import annotations

import numpy as np

from src.tracking.track import Track, TrackState


class TestTrack:
    def _make_track(self, track_id: int = 1, box: list[float] | None = None) -> Track:
        if box is None:
            box = [100.0, 100.0, 200.0, 200.0]
        return Track(
            track_id=track_id,
            box=np.array(box),
            score=0.9,
            class_id=0,
            class_name="person",
        )

    def test_creation(self) -> None:
        track = self._make_track()
        assert track.track_id == 1
        assert track.score == 0.9
        assert track.class_id == 0
        assert track.class_name == "person"
        assert track.state == TrackState.TENTATIVE

    def test_center(self) -> None:
        track = self._make_track(box=[100, 100, 200, 200])
        cx, cy = track.center
        assert cx == 150.0
        assert cy == 150.0

    def test_dimensions(self) -> None:
        track = self._make_track(box=[0, 0, 100, 50])
        assert track.width == 100.0
        assert track.height == 50.0
        assert track.area == 5000.0

    def test_aspect_ratio(self) -> None:
        track = self._make_track(box=[0, 0, 100, 50])
        assert track.aspect_ratio == 2.0

    def test_is_confirmed(self) -> None:
        track = self._make_track()
        assert not track.is_confirmed
        track.state = TrackState.CONFIRMED
        assert track.is_confirmed

    def test_is_deleted(self) -> None:
        track = self._make_track()
        assert not track.is_deleted
        track.state = TrackState.DELETED
        assert track.is_deleted

    def test_update_box(self) -> None:
        track = self._make_track(box=[100, 100, 200, 200])
        track.update_box(
            np.array([110, 110, 210, 210]),
            score=0.85,
            class_id=0,
            class_name="person",
        )
        assert track.hits == 1
        assert track.time_since_update == 0
        assert track.age == 1
        assert len(track.trajectory) == 1

    def test_mark_missed(self) -> None:
        track = self._make_track()
        track.mark_missed()
        assert track.time_since_update == 1
        assert track.age == 1

    def test_predict_next_position(self) -> None:
        track = self._make_track(box=[100, 100, 200, 200])
        track.velocity = np.array([10.0, 5.0])
        predicted = track.predict_next_position()
        assert predicted[0] == 110.0
        assert predicted[1] == 105.0

    def test_add_feature(self) -> None:
        track = self._make_track()
        feat = np.random.randn(128)
        track.add_feature(feat)
        assert len(track.features) == 1
        assert track.get_latest_feature() is not None

    def test_feature_budget(self) -> None:
        track = self._make_track()
        for i in range(150):
            track.add_feature(np.random.randn(128), max_features=100)
        assert len(track.features) == 100

    def test_to_tlwh(self) -> None:
        track = self._make_track(box=[10, 20, 110, 70])
        tlwh = track.to_tlwh()
        assert tlwh[0] == 10
        assert tlwh[1] == 20
        assert tlwh[2] == 100
        assert tlwh[3] == 50

    def test_to_xyah(self) -> None:
        track = self._make_track(box=[0, 0, 100, 50])
        xyah = track.to_xyah()
        assert xyah[0] == 50.0
        assert xyah[1] == 25.0
        assert xyah[2] == 2.0
        assert xyah[3] == 50.0

    def test_speed_and_direction(self) -> None:
        track = self._make_track()
        track.velocity = np.array([3.0, 4.0])
        assert track.speed == 5.0
        assert track.direction != 0.0

    def test_to_dict(self) -> None:
        track = self._make_track()
        d = track.to_dict()
        assert "track_id" in d
        assert "box" in d
        assert "state" in d
        assert "velocity" in d
        assert d["track_id"] == 1

    def test_trajectory_array(self) -> None:
        track = self._make_track()
        track.trajectory.append((150.0, 150.0))
        track.trajectory.append((160.0, 155.0))
        arr = track.get_trajectory_array()
        assert arr.shape == (2, 2)

    def test_empty_trajectory_array(self) -> None:
        track = self._make_track()
        arr = track.get_trajectory_array()
        assert arr.shape == (0, 2)
