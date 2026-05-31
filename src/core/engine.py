from __future__ import annotations

import signal
import time
from typing import Any

import cv2

from src.analytics.counter import ObjectCounter
from src.analytics.dwell_time import DwellTimeAnalyzer
from src.analytics.event_detector import EventDetector
from src.analytics.speed_estimator import SpeedEstimator
from src.analytics.zone import ZoneAnalytics
from src.capture.stream_manager import StreamManager
from src.config.settings import Settings
from src.config.validator import ConfigValidator
from src.core.exceptions import PipelineError, PipelineStageError
from src.detection.base_detector import BaseObjectDetector
from src.detection.detector_factory import DetectorFactory
from src.io.csv_exporter import CSVExporter
from src.io.json_exporter import JSONExporter
from src.io.video_writer import VideoWriter
from src.metrics.metrics_aggregator import MetricsAggregator
from src.tracking.base_tracker import BaseObjectTracker
from src.tracking.tracker_factory import TrackerFactory
from src.utils.logger import get_logger, setup_logging
from src.visualization.heatmap import DetectionHeatmap
from src.visualization.hud import HeadsUpDisplay
from src.visualization.renderer import OverlayRenderer
from src.visualization.trajectory import TrajectoryVisualizer

logger = get_logger(__name__)


class PipelineEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._stream_manager: StreamManager | None = None
        self._detector: BaseObjectDetector | None = None
        self._tracker: BaseObjectTracker | None = None
        self._renderer: OverlayRenderer | None = None
        self._hud: HeadsUpDisplay | None = None
        self._trajectory_viz: TrajectoryVisualizer | None = None
        self._heatmap: DetectionHeatmap | None = None
        self._counter: ObjectCounter | None = None
        self._zone_analytics: ZoneAnalytics | None = None
        self._speed_estimator: SpeedEstimator | None = None
        self._dwell_analyzer: DwellTimeAnalyzer | None = None
        self._event_detector: EventDetector | None = None
        self._metrics: MetricsAggregator | None = None
        self._video_writer: VideoWriter | None = None
        self._json_exporter: JSONExporter | None = None
        self._csv_exporter: CSVExporter | None = None
        self._running = False
        self._paused = False
        self._frame_count: int = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def initialize(self) -> None:
        logger.info("initializing_pipeline", project=self._settings.project_name)

        setup_logging(
            level=self._settings.logging.level,
            log_format=self._settings.logging.format,
            log_file=self._settings.logging.file,
        )

        validator = ConfigValidator(self._settings)
        validator.validate()
        if validator.warnings:
            for w in validator.warnings:
                logger.warning("config_warning", warning=w)

        self._stream_manager = StreamManager(self._settings.source)
        source = self._stream_manager.create_source_from_settings()
        self._stream_manager.initialize_active()

        self._detector = DetectorFactory.create_from_settings(self._settings.model)
        self._detector.initialize()
        if self._settings.model.warmup_iterations > 0:
            self._detector.warmup(self._settings.model.warmup_iterations)

        self._tracker = TrackerFactory.create_from_settings(self._settings.tracker)
        self._tracker.initialize()

        vis = self._settings.visualization
        if vis.enabled:
            self._renderer = OverlayRenderer(
                show_boxes=vis.show_boxes,
                show_labels=vis.show_labels,
                show_confidence=vis.show_confidence,
                show_tracks=vis.show_tracks,
                show_trajectory=vis.show_trajectory,
                trajectory_length=vis.trajectory_length,
                box_thickness=vis.box_thickness,
                font_scale=vis.font_scale,
                font_thickness=vis.font_thickness,
            )
            self._hud = HeadsUpDisplay()
            self._trajectory_viz = TrajectoryVisualizer(max_length=vis.trajectory_length)

        resolution = (
            self._stream_manager.active_source.resolution if self._stream_manager.active_source else (1280, 720)
        )
        self._heatmap = DetectionHeatmap(width=resolution[0], height=resolution[1])

        analytics = self._settings.analytics
        if analytics.counting.enabled:
            self._counter = ObjectCounter(
                line_start=analytics.counting.line_start,
                line_end=analytics.counting.line_end,
            )

        if analytics.zones.enabled and analytics.zones.regions:
            self._zone_analytics = ZoneAnalytics()

        if analytics.speed_estimation.enabled:
            self._speed_estimator = SpeedEstimator(
                pixels_per_meter=analytics.speed_estimation.pixels_per_meter,
            )

        if analytics.dwell_time.enabled:
            self._dwell_analyzer = DwellTimeAnalyzer(
                threshold_seconds=analytics.dwell_time.threshold_seconds,
            )

        self._event_detector = EventDetector()
        self._metrics = MetricsAggregator()
        self._metrics.fps_counter.start()

        output = self._settings.output
        if output.save_video:
            self._video_writer = VideoWriter(
                output_path=output.video_path,
                fps=output.video_fps,
                codec=output.video_codec,
            )

        if output.save_json:
            self._json_exporter = JSONExporter(output_path=output.json_path)

        if output.save_csv:
            self._csv_exporter = CSVExporter(output_path=output.csv_path)

        logger.info("pipeline_initialized")

    def run(self) -> None:
        self._running = True
        logger.info("pipeline_started")

        signal.signal(signal.SIGINT, self._signal_handler)

        try:
            while self._running:
                if self._paused:
                    time.sleep(0.1)
                    continue

                success = self._process_frame()
                if not success:
                    logger.info("source_ended")
                    break

        except KeyboardInterrupt:
            logger.info("pipeline_interrupted")
        except Exception as e:
            logger.error("pipeline_error", error=str(e), error_type=type(e).__name__)
            raise PipelineError(str(e))
        finally:
            self.shutdown()

    def _process_frame(self) -> bool:
        if self._stream_manager is None:
            return False

        self._metrics.start_stage("capture")
        ret, frame = self._stream_manager.read_active()
        self._metrics.stop_stage("capture")

        if not ret or frame is None:
            return False

        self._frame_count += 1

        self._metrics.start_stage("detection")
        try:
            detections = self._detector.detect(frame)
        except Exception as e:
            raise PipelineStageError("detection", e)
        self._metrics.stop_stage("detection")

        self._metrics.start_stage("tracking")
        try:
            tracks = self._tracker.update(detections, frame)
        except Exception as e:
            raise PipelineStageError("tracking", e)
        self._metrics.stop_stage("tracking")

        self._metrics.start_stage("analytics")
        speeds = {}
        if self._counter:
            self._counter.update(tracks)
        if self._zone_analytics:
            self._zone_analytics.update(tracks)
        if self._speed_estimator:
            speeds = self._speed_estimator.update(tracks)
        if self._dwell_analyzer:
            self._dwell_analyzer.update(tracks)
        if self._event_detector:
            self._event_detector.update(tracks, speeds)
        if self._heatmap:
            self._heatmap.update(detections)
        self._metrics.stop_stage("analytics")

        self._metrics.start_stage("visualization")
        display_frame = frame.copy()

        if self._renderer:
            display_frame = self._renderer.draw_tracks(display_frame, tracks)

            if self._counter:
                display_frame = self._renderer.draw_counting_line(
                    display_frame,
                    self._counter.line_start,
                    self._counter.line_end,
                    count_in=self._counter.count_in,
                    count_out=self._counter.count_out,
                )

        fps = self._metrics.tick()

        if self._hud:
            source = self._stream_manager.active_source
            resolution_str = f"{source.resolution[0]}x{source.resolution[1]}" if source else ""
            latency_ms = self._metrics.latency_tracker.get_total_latency_ms()

            display_frame = self._hud.render(
                display_frame,
                fps=fps,
                latency_ms=latency_ms,
                num_detections=detections.num_detections,
                num_tracks=len(tracks),
                frame_id=self._frame_count,
                model_name=self._settings.model.name,
                tracker_name=self._settings.tracker.type,
                resolution=resolution_str,
            )

        self._metrics.stop_stage("visualization")

        self._metrics.start_stage("export")
        if self._video_writer:
            self._video_writer.write(display_frame)

        if self._json_exporter:
            self._json_exporter.add_frame_result(
                frame_id=self._frame_count,
                timestamp=time.time(),
                detections=detections.to_dict()["detections"],
                tracks=[t.to_dict() for t in tracks],
            )

        if self._csv_exporter:
            self._csv_exporter.add_tracks(
                frame_id=self._frame_count,
                timestamp=time.time(),
                tracks=[t.to_dict() for t in tracks],
            )
        self._metrics.stop_stage("export")

        cv2.imshow("ObjaTrack-XL", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            self._running = False
        elif key == ord("p"):
            self._paused = not self._paused
        elif key == ord("h"):
            if self._heatmap:
                display_frame = self._heatmap.render(display_frame)
                cv2.imshow("ObjaTrack-XL Heatmap", display_frame)

        self._metrics.increment_counter("total_detections", detections.num_detections)
        self._metrics.increment_counter("total_tracks", len(tracks))

        if self._frame_count % 300 == 0:
            self._metrics.log_summary()

        return True

    def pause(self) -> None:
        self._paused = True
        logger.info("pipeline_paused")

    def resume(self) -> None:
        self._paused = False
        logger.info("pipeline_resumed")

    def shutdown(self) -> None:
        logger.info("shutting_down_pipeline")
        self._running = False

        if self._stream_manager:
            self._stream_manager.release_all()

        if self._detector:
            self._detector.shutdown()

        if self._tracker:
            self._tracker.shutdown()

        if self._video_writer:
            self._video_writer.release()

        if self._json_exporter:
            self._json_exporter.export()

        if self._csv_exporter:
            self._csv_exporter.export()

        cv2.destroyAllWindows()

        if self._metrics:
            summary = self._metrics.get_snapshot()
            logger.info("pipeline_final_metrics", **self._metrics.get_summary())

        logger.info("pipeline_shutdown_complete", total_frames=self._frame_count)

    def get_metrics_snapshot(self) -> dict[str, Any]:
        if self._metrics:
            return self._metrics.get_snapshot()
        return {}

    def _signal_handler(self, signum: int, frame: Any) -> None:
        logger.info("received_signal", signal=signum)
        self._running = False
