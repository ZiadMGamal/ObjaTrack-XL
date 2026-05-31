from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelProfiler:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path

    def profile_onnx(self) -> dict[str, Any]:
        try:
            import onnx

            model = onnx.load(self._model_path)
            graph = model.graph

            op_counts: dict[str, int] = {}
            for node in graph.node:
                op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

            total_params = 0
            initializer_info = []
            for init in graph.initializer:
                shape = list(init.dims)
                params = 1
                for d in shape:
                    params *= d
                total_params += params
                initializer_info.append(
                    {
                        "name": init.name,
                        "shape": shape,
                        "params": params,
                        "dtype": str(init.data_type),
                    }
                )

            return {
                "model": Path(self._model_path).name,
                "total_nodes": len(graph.node),
                "total_params": total_params,
                "total_params_millions": round(total_params / 1e6, 2),
                "op_counts": dict(sorted(op_counts.items(), key=lambda x: x[1], reverse=True)),
                "num_inputs": len(graph.input),
                "num_outputs": len(graph.output),
                "num_initializers": len(graph.initializer),
                "top_initializers": sorted(initializer_info, key=lambda x: x["params"], reverse=True)[:20],
            }
        except Exception as e:
            return {"error": str(e)}

    def profile_runtime(
        self,
        iterations: int = 100,
        warmup: int = 10,
        providers: list[str] | None = None,
    ) -> dict[str, Any]:
        import time

        import onnxruntime as ort

        providers = providers or ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = True

        session = ort.InferenceSession(self._model_path, sess_options=sess_options, providers=providers)

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

        profile_file = session.end_profiling()

        latencies_arr = np.array(latencies)

        return {
            "model": Path(self._model_path).name,
            "providers": providers,
            "iterations": iterations,
            "latency_stats": {
                "mean_ms": round(float(np.mean(latencies_arr)), 3),
                "std_ms": round(float(np.std(latencies_arr)), 3),
                "min_ms": round(float(np.min(latencies_arr)), 3),
                "max_ms": round(float(np.max(latencies_arr)), 3),
                "p50_ms": round(float(np.percentile(latencies_arr, 50)), 3),
                "p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
                "p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
            },
            "throughput_fps": round(1000.0 / float(np.mean(latencies_arr)), 1),
            "profile_file": profile_file,
        }

    def estimate_flops(self) -> dict[str, Any]:
        try:
            import onnx

            model = onnx.load(self._model_path)

            total_flops = 0
            conv_flops = 0
            matmul_flops = 0

            for node in model.graph.node:
                if node.op_type == "Conv":
                    conv_flops += self._estimate_conv_flops(node, model.graph)
                elif node.op_type in ("MatMul", "Gemm"):
                    matmul_flops += self._estimate_matmul_flops(node, model.graph)

            total_flops = conv_flops + matmul_flops

            return {
                "total_gflops": round(total_flops / 1e9, 2),
                "conv_gflops": round(conv_flops / 1e9, 2),
                "matmul_gflops": round(matmul_flops / 1e9, 2),
            }
        except Exception as e:
            return {"error": str(e)}

    def _estimate_conv_flops(self, node: Any, graph: Any) -> int:
        for init in graph.initializer:
            if init.name == node.input[1]:
                shape = list(init.dims)
                if len(shape) == 4:
                    out_channels, in_channels, kh, kw = shape
                    return 2 * out_channels * in_channels * kh * kw
        return 0

    def _estimate_matmul_flops(self, node: Any, graph: Any) -> int:
        for init in graph.initializer:
            if init.name in node.input:
                shape = list(init.dims)
                if len(shape) == 2:
                    return 2 * shape[0] * shape[1]
        return 0
