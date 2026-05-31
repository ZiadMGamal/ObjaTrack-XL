from __future__ import annotations

from typing import Any

from src.core.exceptions import TrackingError
from src.config.settings import TrackerSettings
from src.tracking.base_tracker import BaseObjectTracker
from src.tracking.sort_tracker import SORTTracker
from src.tracking.byte_tracker import ByteTrackTracker
from src.tracking.bot_sort_tracker import BoTSORTTracker
from src.tracking.deep_sort_tracker import DeepSORTTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrackerFactory:

    _tracker_map: dict[str, type[BaseObjectTracker]] = {
        "sort": SORTTracker,
        "bytetrack": ByteTrackTracker,
        "botsort": BoTSORTTracker,
        "deepsort": DeepSORTTracker,
    }

    @classmethod
    def create(cls, tracker_type: str, **kwargs: Any) -> BaseObjectTracker:
        tracker_type = tracker_type.lower()

        if tracker_type not in cls._tracker_map:
            available = list(cls._tracker_map.keys())
            raise TrackingError(
                f"Unknown tracker type: {tracker_type}. Available: {available}",
                tracker_type=tracker_type,
            )

        tracker_cls = cls._tracker_map[tracker_type]
        logger.info("creating_tracker", type=tracker_type)
        return tracker_cls(**kwargs)

    @classmethod
    def create_from_settings(cls, settings: TrackerSettings) -> BaseObjectTracker:
        tracker_type = settings.type.lower()
        common_kwargs: dict[str, Any] = {
            "max_age": settings.max_age,
            "min_hits": settings.min_hits,
            "iou_threshold": settings.iou_threshold,
        }

        if tracker_type in ("bytetrack", "botsort"):
            common_kwargs["track_high_thresh"] = settings.track_high_thresh
            common_kwargs["track_low_thresh"] = settings.track_low_thresh
            common_kwargs["new_track_thresh"] = settings.new_track_thresh
            common_kwargs["match_thresh"] = settings.match_thresh

        if tracker_type == "botsort":
            common_kwargs["with_reid"] = settings.with_reid

        if tracker_type == "deepsort":
            common_kwargs["max_cosine_distance"] = settings.max_cosine_distance
            common_kwargs["nn_budget"] = settings.nn_budget

        return cls.create(tracker_type, **common_kwargs)

    @classmethod
    def register_tracker(cls, name: str, tracker_cls: type[BaseObjectTracker]) -> None:
        cls._tracker_map[name.lower()] = tracker_cls
        logger.info("tracker_registered", name=name)

    @classmethod
    def available_trackers(cls) -> list[str]:
        return list(cls._tracker_map.keys())
