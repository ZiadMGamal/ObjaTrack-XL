from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelValidator:

    def __init__(self, tolerance: float = 0.01, sample_count: int = 50) -> None:
        self._tolerance = tolerance
        self._sample_count = sample_count

    def validate_onnx(self, model_path: str) -> dict[str, Any]:
        try:
            import onnx
            model = onnx.load(model_path)
            onnx.checker.check_model(model)

            graph = model.graph
            return {
                "valid": True,
                "format": "onnx",
                "path": model_path,
                "size_mb": round(Path(model_path).stat().st_size / (1024 * 1024), 2),
                "opset": model.opset_import[0].version if model.opset_import else None,
                "num_nodes": len(graph.node),
                "num_inputs": len(graph.input),
                "num_outputs": len(graph.output),
                "input_shapes": {
                    inp.name: [d.dim_value for d in inp.type.tensor_type.shape.dim]
                    for inp in graph.input
                },
                "output_shapes": {
                    out.name: [d.dim_value for d in out.type.tensor_type.shape.dim]
                    for out in graph.output
                },
                "op_types": list(set(n.op_type for n in graph.node)),
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def compare_outputs(
        self,
        original_path: str,
        optimized_path: str,
        input_size: tuple[int, int] = (640, 640),
    ) -> dict[str, Any]:
        try:
            import onnxruntime as ort

            orig = ort.InferenceSession(original_path, providers=["CPUExecutionProvider"])
            opt = ort.InferenceSession(optimized_path, providers=["CPUExecutionProvider"])

            orig_input = orig.get_inputs()[0]
            opt_input = opt.get_inputs()[0]

            differences = []
            cosine_sims = []

            for _ in range(self._sample_count):
                shape = [s if isinstance(s, int) else 1 for s in orig_input.shape]
                dummy = np.random.randn(*shape).astype(np.float32)

                orig_out = orig.run(None, {orig_input.name: dummy})[0].flatten()
                opt_out = opt.run(None, {opt_input.name: dummy})[0].flatten()

                diff = np.abs(orig_out - opt_out)
                differences.append(float(np.mean(diff)))

                norm_orig = np.linalg.norm(orig_out)
                norm_opt = np.linalg.norm(opt_out)
                if norm_orig > 0 and norm_opt > 0:
                    cosine_sims.append(float(np.dot(orig_out, opt_out) / (norm_orig * norm_opt)))

            avg_diff = sum(differences) / len(differences)
            avg_cosine = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0

            return {
                "valid": avg_diff <= self._tolerance,
                "samples": self._sample_count,
                "avg_absolute_diff": round(avg_diff, 6),
                "max_absolute_diff": round(max(differences), 6),
                "min_absolute_diff": round(min(differences), 6),
                "avg_cosine_similarity": round(avg_cosine, 6),
                "tolerance": self._tolerance,
                "within_tolerance": avg_diff <= self._tolerance,
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def benchmark_inference(
        self,
        model_path: str,
        input_size: tuple[int, int] = (640, 640),
        iterations: int = 100,
        warmup: int = 10,
        providers: list[str] | None = None,
    ) -> dict[str, Any]:
        import time
        import onnxruntime as ort

        providers = providers or ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        input_info = session.get_inputs()[0]
        shape = [s if isinstance(s, int) else 1 for s in input_info.shape]
        dummy = np.random.randn(*shape).astype(np.float32)

        for _ in range(warmup):
            session.run(None, {input_info.name: dummy})

        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            session.run(None, {input_info.name: dummy})
            latencies.append((time.perf_counter() - start) * 1000)

        latencies_arr = np.array(latencies)
        return {
            "model": Path(model_path).name,
            "iterations": iterations,
            "avg_latency_ms": round(float(np.mean(latencies_arr)), 3),
            "std_latency_ms": round(float(np.std(latencies_arr)), 3),
            "min_latency_ms": round(float(np.min(latencies_arr)), 3),
            "max_latency_ms": round(float(np.max(latencies_arr)), 3),
            "p50_latency_ms": round(float(np.percentile(latencies_arr, 50)), 3),
            "p90_latency_ms": round(float(np.percentile(latencies_arr, 90)), 3),
            "p95_latency_ms": round(float(np.percentile(latencies_arr, 95)), 3),
            "p99_latency_ms": round(float(np.percentile(latencies_arr, 99)), 3),
            "avg_fps": round(1000.0 / float(np.mean(latencies_arr)), 1),
            "providers": providers,
        }
