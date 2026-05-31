from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DeviceInfo:
    device_type: str = "cpu"
    device_index: int = 0
    device_name: str = ""
    total_memory_mb: float = 0.0
    compute_capability: tuple[int, int] = (0, 0)
    cuda_version: str = ""
    cudnn_version: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


class DeviceManager:

    _instance: DeviceManager | None = None
    _device_info: DeviceInfo | None = None

    def __new__(cls) -> DeviceManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._device_info is None:
            self._device_info = DeviceInfo()
            self._detect_device()

    def _detect_device(self) -> None:
        if torch.cuda.is_available():
            self._device_info.device_type = "cuda"
            self._device_info.device_index = torch.cuda.current_device()
            self._device_info.device_name = torch.cuda.get_device_name(self._device_info.device_index)
            props = torch.cuda.get_device_properties(self._device_info.device_index)
            self._device_info.total_memory_mb = props.total_mem / (1024 * 1024)
            self._device_info.compute_capability = (props.major, props.minor)
            self._device_info.cuda_version = torch.version.cuda or ""
            self._device_info.cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else ""
            self._device_info.properties = {
                "multi_processor_count": props.multi_processor_count,
                "max_threads_per_multi_processor": props.max_threads_per_multi_processor,
            }
            logger.info(
                "device_detected",
                device=self._device_info.device_name,
                memory_mb=round(self._device_info.total_memory_mb, 1),
                cuda=self._device_info.cuda_version,
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device_info.device_type = "mps"
            self._device_info.device_name = "Apple MPS"
            logger.info("device_detected", device="Apple MPS")
        else:
            self._device_info.device_type = "cpu"
            self._device_info.device_name = platform.processor() or "CPU"
            logger.info("device_detected", device="CPU")

    @property
    def device(self) -> torch.device:
        if self._device_info.device_type == "cuda":
            return torch.device(f"cuda:{self._device_info.device_index}")
        return torch.device(self._device_info.device_type)

    @property
    def device_type(self) -> str:
        return self._device_info.device_type

    @property
    def is_cuda(self) -> bool:
        return self._device_info.device_type == "cuda"

    @property
    def is_mps(self) -> bool:
        return self._device_info.device_type == "mps"

    @property
    def is_cpu(self) -> bool:
        return self._device_info.device_type == "cpu"

    @property
    def device_info(self) -> DeviceInfo:
        return self._device_info

    def get_device_string(self) -> str:
        if self._device_info.device_type == "cuda":
            return f"cuda:{self._device_info.device_index}"
        return self._device_info.device_type

    def get_onnx_providers(self) -> list[str]:
        providers = []
        if self.is_cuda:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def get_memory_usage(self) -> dict[str, float]:
        if not self.is_cuda:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
        return {
            "allocated_mb": torch.cuda.memory_allocated(self._device_info.device_index) / (1024 * 1024),
            "reserved_mb": torch.cuda.memory_reserved(self._device_info.device_index) / (1024 * 1024),
            "max_allocated_mb": torch.cuda.max_memory_allocated(self._device_info.device_index) / (1024 * 1024),
        }

    def clear_cache(self) -> None:
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self._device_info.device_index)

    def synchronize(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize(self._device_info.device_index)

    def get_system_info(self) -> dict[str, Any]:
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device_type": self._device_info.device_type,
            "device_name": self._device_info.device_name,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": self._device_info.cuda_version,
            "cudnn_version": self._device_info.cudnn_version,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "total_memory_mb": round(self._device_info.total_memory_mb, 1),
        }

    def __repr__(self) -> str:
        return f"DeviceManager(device={self.get_device_string()!r}, name={self._device_info.device_name!r})"
