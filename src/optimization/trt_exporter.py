from __future__ import annotations

from src.core.base import ComponentState
from src.core.exceptions import OptimizationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TensorRTExporter:

    def __init__(
        self,
        model_path: str,
        output_dir: str = "models/optimized",
        workspace_size: int = 4294967296,
        fp16_mode: bool = True,
        int8_mode: bool = False,
        max_batch_size: int = 1,
    ) -> None:
        self._model_path = model_path
        self._output_dir = output_dir
        self._workspace_size = workspace_size
        self._fp16_mode = fp16_mode
        self._int8_mode = int8_mode
        self._max_batch_size = max_batch_size

    def export(self) -> str:
        logger.info("tensorrt_export_placeholder", model=self._model_path)
        try:
            from ultralytics import YOLO
            model = YOLO(self._model_path)
            result = model.export(
                format="engine",
                half=self._fp16_mode,
                int8=self._int8_mode,
                workspace=self._workspace_size // (1024 * 1024 * 1024),
                batch=self._max_batch_size,
            )
            logger.info("tensorrt_export_complete", path=str(result))
            return str(result)
        except Exception as e:
            raise OptimizationError(f"TensorRT export failed: {e}", format_type="tensorrt")
