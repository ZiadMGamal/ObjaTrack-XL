from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.core.base import ComponentState
from src.core.exceptions import QuantizationError
from src.optimization.base_optimizer import BaseModelOptimizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelQuantizer(BaseModelOptimizer):

    def __init__(
        self,
        model_path: str,
        output_dir: str = "models/optimized",
        input_size: tuple[int, int] = (640, 640),
        method: str = "dynamic",
        precision: str = "int8",
        per_channel: bool = True,
        calibration_samples: int = 100,
        name: str | None = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            output_dir=output_dir,
            input_size=input_size,
            name=name or "ModelQuantizer",
        )
        self._method = method
        self._precision = precision
        self._per_channel = per_channel
        self._calibration_samples = calibration_samples

    def initialize(self) -> None:
        self.set_state(ComponentState.READY)
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)

    def optimize(self) -> str:
        if self._method == "dynamic":
            return self._dynamic_quantize()
        elif self._method == "static":
            return self._static_quantize()
        else:
            raise QuantizationError(f"Unknown quantization method: {self._method}")

    def _dynamic_quantize(self) -> str:
        self.set_state(ComponentState.RUNNING)
        start_time = time.time()

        logger.info(
            "starting_dynamic_quantization",
            model=self._model_path,
            precision=self._precision,
        )

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            input_path = self._model_path
            stem = Path(input_path).stem
            output_path = str(Path(self._output_dir) / f"{stem}_{self._precision}.onnx")

            quant_type = QuantType.QInt8 if self._precision == "int8" else QuantType.QUInt8

            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                per_channel=self._per_channel,
                weight_type=quant_type,
            )

        except ImportError:
            self.set_state(ComponentState.ERROR)
            raise QuantizationError("onnxruntime.quantization not available")
        except Exception as e:
            self.set_state(ComponentState.ERROR)
            raise QuantizationError(str(e), precision=self._precision)

        elapsed = time.time() - start_time
        size_info = self.compute_size_reduction(self._model_path, output_path)

        self._optimization_results = {
            "output_path": output_path,
            "method": "dynamic",
            "precision": self._precision,
            "quantization_time_seconds": round(elapsed, 2),
            "per_channel": self._per_channel,
            **size_info,
        }

        self.set_state(ComponentState.READY)
        logger.info("quantization_complete", **self._optimization_results)
        return output_path

    def _static_quantize(self) -> str:
        self.set_state(ComponentState.RUNNING)
        start_time = time.time()

        logger.info(
            "starting_static_quantization",
            model=self._model_path,
            calibration_samples=self._calibration_samples,
        )

        try:
            from onnxruntime.quantization import (
                quantize_static,
                CalibrationDataReader,
                QuantType,
                QuantFormat,
            )

            class RandomCalibrationReader(CalibrationDataReader):
                def __init__(self, input_size: tuple[int, int], num_samples: int):
                    self._data = iter([
                        {"images": np.random.randn(1, 3, *input_size).astype(np.float32)}
                        for _ in range(num_samples)
                    ])

                def get_next(self) -> dict | None:
                    try:
                        return next(self._data)
                    except StopIteration:
                        return None

            input_path = self._model_path
            stem = Path(input_path).stem
            output_path = str(Path(self._output_dir) / f"{stem}_{self._precision}_static.onnx")

            calibration_reader = RandomCalibrationReader(
                self._input_size, self._calibration_samples
            )

            quantize_static(
                model_input=input_path,
                model_output=output_path,
                calibration_data_reader=calibration_reader,
                quant_format=QuantFormat.QDQ,
                per_channel=self._per_channel,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QInt8,
            )

        except ImportError:
            self.set_state(ComponentState.ERROR)
            raise QuantizationError("onnxruntime.quantization not available")
        except Exception as e:
            self.set_state(ComponentState.ERROR)
            raise QuantizationError(str(e), precision=self._precision)

        elapsed = time.time() - start_time
        size_info = self.compute_size_reduction(self._model_path, output_path)

        self._optimization_results = {
            "output_path": output_path,
            "method": "static",
            "precision": self._precision,
            "calibration_samples": self._calibration_samples,
            "quantization_time_seconds": round(elapsed, 2),
            **size_info,
        }

        self.set_state(ComponentState.READY)
        logger.info("static_quantization_complete", **self._optimization_results)
        return output_path

    def validate(self, original_path: str, optimized_path: str) -> dict[str, Any]:
        try:
            import onnxruntime as ort

            orig_session = ort.InferenceSession(original_path, providers=["CPUExecutionProvider"])
            quant_session = ort.InferenceSession(optimized_path, providers=["CPUExecutionProvider"])

            input_name = orig_session.get_inputs()[0].name
            input_shape = orig_session.get_inputs()[0].shape
            shape = [s if isinstance(s, int) else 1 for s in input_shape]
            dummy = np.random.randn(*shape).astype(np.float32)

            orig_output = orig_session.run(None, {input_name: dummy})[0]

            quant_input_name = quant_session.get_inputs()[0].name
            quant_output = quant_session.run(None, {quant_input_name: dummy})[0]

            abs_diff = np.abs(orig_output.flatten() - quant_output.flatten())

            return {
                "valid": True,
                "max_abs_diff": float(np.max(abs_diff)),
                "mean_abs_diff": float(np.mean(abs_diff)),
                "median_abs_diff": float(np.median(abs_diff)),
                "cosine_similarity": float(
                    np.dot(orig_output.flatten(), quant_output.flatten()) /
                    (np.linalg.norm(orig_output.flatten()) * np.linalg.norm(quant_output.flatten()) + 1e-8)
                ),
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def shutdown(self) -> None:
        self.set_state(ComponentState.STOPPED)
