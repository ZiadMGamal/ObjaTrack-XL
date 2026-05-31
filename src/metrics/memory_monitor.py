from __future__ import annotations

import os
from typing import Any

import psutil


class MemoryMonitor:
    def __init__(self, enable_gpu: bool = False) -> None:
        self._enable_gpu = enable_gpu
        self._process = psutil.Process(os.getpid())
        self._gpu_available = False

        if enable_gpu:
            try:
                import GPUtil

                self._gputil = GPUtil
                self._gpu_available = True
            except ImportError:
                self._gpu_available = False

    def get_cpu_memory(self) -> dict[str, float]:
        mem_info = self._process.memory_info()
        virtual = psutil.virtual_memory()

        return {
            "rss_mb": round(mem_info.rss / (1024 * 1024), 1),
            "vms_mb": round(mem_info.vms / (1024 * 1024), 1),
            "percent": round(self._process.memory_percent(), 1),
            "system_total_mb": round(virtual.total / (1024 * 1024), 1),
            "system_available_mb": round(virtual.available / (1024 * 1024), 1),
            "system_percent": round(virtual.percent, 1),
        }

    def get_gpu_memory(self) -> dict[str, Any]:
        if not self._gpu_available:
            return {"available": False}

        try:
            gpus = self._gputil.getGPUs()
            if not gpus:
                return {"available": False}

            gpu = gpus[0]
            return {
                "available": True,
                "gpu_name": gpu.name,
                "total_mb": round(gpu.memoryTotal, 1),
                "used_mb": round(gpu.memoryUsed, 1),
                "free_mb": round(gpu.memoryFree, 1),
                "utilization": round(gpu.memoryUtil * 100, 1),
                "gpu_load": round(gpu.load * 100, 1),
                "temperature": gpu.temperature,
            }
        except Exception:
            return {"available": False}

    def get_all_memory(self) -> dict[str, Any]:
        result = {"cpu": self.get_cpu_memory()}
        if self._enable_gpu:
            result["gpu"] = self.get_gpu_memory()
        return result

    def get_torch_memory(self) -> dict[str, float]:
        try:
            import torch

            if not torch.cuda.is_available():
                return {"available": False}

            return {
                "available": True,
                "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 1),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1),
            }
        except ImportError:
            return {"available": False}
