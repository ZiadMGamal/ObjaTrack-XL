from __future__ import annotations

import time
import functools
from typing import Any, Callable, TypeVar
from contextlib import ContextDecorator

from src.utils.logger import get_logger

F = TypeVar("F", bound=Callable[..., Any])

logger = get_logger(__name__)


class Timer(ContextDecorator):

    def __init__(self, name: str = "operation", log_output: bool = False) -> None:
        self._name = name
        self._log_output = log_output
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._elapsed: float = 0.0

    @property
    def elapsed(self) -> float:
        return self._elapsed

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed * 1000.0

    def __enter__(self) -> Timer:
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        self._end_time = time.perf_counter()
        self._elapsed = self._end_time - self._start_time
        if self._log_output:
            logger.info(
                "timer_completed",
                operation=self._name,
                elapsed_ms=round(self.elapsed_ms, 3),
            )

    def reset(self) -> None:
        self._start_time = 0.0
        self._end_time = 0.0
        self._elapsed = 0.0

    def __repr__(self) -> str:
        return f"Timer(name={self._name!r}, elapsed_ms={self.elapsed_ms:.3f})"


class AccumulatingTimer:

    def __init__(self, name: str = "accumulator") -> None:
        self._name = name
        self._total_time: float = 0.0
        self._count: int = 0
        self._min_time: float = float("inf")
        self._max_time: float = 0.0
        self._times: list[float] = []

    @property
    def total_time(self) -> float:
        return self._total_time

    @property
    def count(self) -> int:
        return self._count

    @property
    def average_time(self) -> float:
        return self._total_time / self._count if self._count > 0 else 0.0

    @property
    def average_time_ms(self) -> float:
        return self.average_time * 1000.0

    @property
    def min_time_ms(self) -> float:
        return self._min_time * 1000.0 if self._min_time != float("inf") else 0.0

    @property
    def max_time_ms(self) -> float:
        return self._max_time * 1000.0

    def measure(self) -> Timer:
        timer = Timer(name=self._name)
        self._current_timer = timer
        return _AccumulatingTimerContext(self, timer)

    def record(self, elapsed: float) -> None:
        self._total_time += elapsed
        self._count += 1
        self._min_time = min(self._min_time, elapsed)
        self._max_time = max(self._max_time, elapsed)
        self._times.append(elapsed)

    def percentile(self, p: float) -> float:
        if not self._times:
            return 0.0
        sorted_times = sorted(self._times)
        idx = int(len(sorted_times) * p / 100.0)
        idx = min(idx, len(sorted_times) - 1)
        return sorted_times[idx] * 1000.0

    def get_statistics(self) -> dict[str, float]:
        return {
            "total_ms": round(self._total_time * 1000.0, 3),
            "count": self._count,
            "average_ms": round(self.average_time_ms, 3),
            "min_ms": round(self.min_time_ms, 3),
            "max_ms": round(self.max_time_ms, 3),
            "p50_ms": round(self.percentile(50), 3),
            "p90_ms": round(self.percentile(90), 3),
            "p95_ms": round(self.percentile(95), 3),
            "p99_ms": round(self.percentile(99), 3),
        }

    def reset(self) -> None:
        self._total_time = 0.0
        self._count = 0
        self._min_time = float("inf")
        self._max_time = 0.0
        self._times.clear()


class _AccumulatingTimerContext:

    def __init__(self, accumulator: AccumulatingTimer, timer: Timer) -> None:
        self._accumulator = accumulator
        self._timer = timer

    def __enter__(self) -> Timer:
        self._timer.__enter__()
        return self._timer

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        self._timer.__exit__(exc_type, exc_val, exc_tb)
        self._accumulator.record(self._timer.elapsed)


def timeit(name: str | None = None, log_output: bool = True) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        op_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Timer(op_name, log_output=log_output):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
