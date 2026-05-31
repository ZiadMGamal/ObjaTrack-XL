from __future__ import annotations

from src.core.base import ComponentState
from src.core.exceptions import DetectionError
from src.core.registry import detector_registry
from src.detection.base_detector import BaseObjectDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


@detector_registry.register("tensorrt")
class TensorRTDetector(BaseObjectDetector):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._engine = None
        self._context = None

    def initialize(self) -> None:
        self.set_state(ComponentState.INITIALIZING)
        try:
            import tensorrt as trt
            self._trt = trt
            logger.info("tensorrt_available", version=trt.__version__)
        except ImportError:
            self.set_state(ComponentState.ERROR)
            raise DetectionError(
                "TensorRT not available. Install tensorrt package.",
                model_name=self._model_path,
            )

        trt_logger = self._trt.Logger(self._trt.Logger.WARNING)
        runtime = self._trt.Runtime(trt_logger)

        with open(self._model_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            self.set_state(ComponentState.ERROR)
            raise DetectionError("Failed to load TensorRT engine")

        self._context = self._engine.create_execution_context()
        self.set_state(ComponentState.READY)
        logger.info("tensorrt_initialized", model=self._model_path)

    def detect(self, frame):
        raise NotImplementedError("TensorRT inference requires CUDA-specific implementation")

    def warmup(self, iterations: int = 10) -> None:
        logger.info("tensorrt_warmup_placeholder", iterations=iterations)

    def shutdown(self) -> None:
        self._context = None
        self._engine = None
        self.set_state(ComponentState.STOPPED)
