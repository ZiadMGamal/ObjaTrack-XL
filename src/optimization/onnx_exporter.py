from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.core.base import ComponentState
from src.core.exceptions import ExportError, OptimizationError
from src.core.registry import optimizer_registry
from src.optimization.base_optimizer import BaseModelOptimizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


@optimizer_registry.register("onnx")
class ONNXExporter(BaseModelOptimizer):

    def __init__(
        self,
        model_path: str,
        output_dir: str = "models/optimized",
        input_size: tuple[int, int] = (640, 640),
        batch_size: int = 1,
        opset_version: int = 17,
        dynamic_axes: bool = True,
        simplify: bool = True,
        half: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            output_dir=output_dir,
            input_size=input_size,
            batch_size=batch_size,
            name=name or "ONNXExporter",
        )
        self._opset_version = opset_version
        self._dynamic_axes = dynamic_axes
        self._simplify = simplify
        self._half = half

    def initialize(self) -> None:
        self.set_state(ComponentState.READY)
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)
        logger.info("onnx_exporter_initialized", output_dir=self._output_dir)

    def optimize(self) -> str:
        self.set_state(ComponentState.RUNNING)
        start_time = time.time()

        logger.info(
            "starting_onnx_export",
            model=self._model_path,
            opset=self._opset_version,
            dynamic=self._dynamic_axes,
            simplify=self._simplify,
            half=self._half,
        )

        try:
            from ultralytics import YOLO
            model = YOLO(self._model_path)

            output_path = model.export(
                format="onnx",
                opset=self._opset_version,
                dynamic=self._dynamic_axes,
                simplify=self._simplify,
                half=self._half,
                imgsz=self._input_size[0],
            )

            output_path = str(output_path)

        except Exception as e:
            self.set_state(ComponentState.ERROR)
            raise ExportError(str(e), source_format="pt", target_format="onnx")

        elapsed = time.time() - start_time

        if self._simplify:
            output_path = self._simplify_model(output_path)

        size_info = self.compute_size_reduction(self._model_path, output_path)

        self._optimization_results = {
            "output_path": output_path,
            "export_time_seconds": round(elapsed, 2),
            "opset_version": self._opset_version,
            "dynamic_axes": self._dynamic_axes,
            "half_precision": self._half,
            **size_info,
        }

        self.set_state(ComponentState.READY)
        logger.info("onnx_export_complete", **self._optimization_results)

        return output_path

    def _simplify_model(self, model_path: str) -> str:
        try:
            import onnx
            from onnxsim import simplify

            model = onnx.load(model_path)
            simplified_model, check = simplify(model)

            if check:
                onnx.save(simplified_model, model_path)
                logger.info("onnx_simplified", path=model_path)
            else:
                logger.warning("onnx_simplification_check_failed")

        except ImportError:
            logger.warning("onnxsim_not_installed")
        except Exception as e:
            logger.warning("onnx_simplification_failed", error=str(e))

        return model_path

    def validate(self, original_path: str, optimized_path: str) -> dict[str, Any]:
        try:
            import onnx
            model = onnx.load(optimized_path)
            onnx.checker.check_model(model)

            return {
                "valid": True,
                "num_nodes": len(model.graph.node),
                "opset_version": model.opset_import[0].version if model.opset_import else None,
                "inputs": [
                    {"name": inp.name, "shape": [d.dim_value for d in inp.type.tensor_type.shape.dim]}
                    for inp in model.graph.input
                ],
                "outputs": [
                    {"name": out.name, "shape": [d.dim_value for d in out.type.tensor_type.shape.dim]}
                    for out in model.graph.output
                ],
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def shutdown(self) -> None:
        self.set_state(ComponentState.STOPPED)
