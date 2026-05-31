from __future__ import annotations

from abc import abstractmethod
from typing import Any

from src.core.base import BaseOptimizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModelOptimizer(BaseOptimizer):

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        input_size: tuple[int, int] = (640, 640),
        batch_size: int = 1,
        name: str | None = None,
    ) -> None:
        super().__init__(model_path=model_path, output_dir=output_dir, name=name)
        self._input_size = input_size
        self._batch_size = batch_size
        self._optimization_results: dict[str, Any] = {}

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def optimization_results(self) -> dict[str, Any]:
        return self._optimization_results

    @abstractmethod
    def optimize(self) -> str:
        ...

    @abstractmethod
    def validate(self, original_path: str, optimized_path: str) -> dict[str, Any]:
        ...

    def get_model_size_mb(self, path: str) -> float:
        from pathlib import Path
        p = Path(path)
        if p.exists():
            return round(p.stat().st_size / (1024 * 1024), 2)
        return 0.0

    def compute_size_reduction(self, original_path: str, optimized_path: str) -> dict[str, float]:
        orig_size = self.get_model_size_mb(original_path)
        opt_size = self.get_model_size_mb(optimized_path)
        reduction = ((orig_size - opt_size) / orig_size * 100) if orig_size > 0 else 0.0

        return {
            "original_size_mb": orig_size,
            "optimized_size_mb": opt_size,
            "reduction_percent": round(reduction, 1),
            "compression_ratio": round(orig_size / opt_size, 2) if opt_size > 0 else 0.0,
        }
