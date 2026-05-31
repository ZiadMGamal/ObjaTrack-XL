from __future__ import annotations

import threading
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    _instances: dict[str, dict[str, type[T]]] = {}
    _lock = threading.Lock()

    def __init__(self, registry_name: str) -> None:
        self._registry_name = registry_name
        with self._lock:
            if registry_name not in self._instances:
                self._instances[registry_name] = {}

    @property
    def registry_name(self) -> str:
        return self._registry_name

    def register(self, name: str) -> Any:
        def decorator(cls: type[T]) -> type[T]:
            with self._lock:
                if name in self._instances[self._registry_name]:
                    raise ValueError(f"Component '{name}' already registered in '{self._registry_name}'")
                self._instances[self._registry_name][name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        with self._lock:
            if name not in self._instances.get(self._registry_name, {}):
                available = list(self._instances.get(self._registry_name, {}).keys())
                raise KeyError(f"Component '{name}' not found in '{self._registry_name}'. Available: {available}")
            return self._instances[self._registry_name][name]

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        cls = self.get(name)
        return cls(*args, **kwargs)

    def list_registered(self) -> list[str]:
        with self._lock:
            return list(self._instances.get(self._registry_name, {}).keys())

    def is_registered(self, name: str) -> bool:
        with self._lock:
            return name in self._instances.get(self._registry_name, {})

    def unregister(self, name: str) -> None:
        with self._lock:
            if name in self._instances.get(self._registry_name, {}):
                del self._instances[self._registry_name][name]

    def clear(self) -> None:
        with self._lock:
            if self._registry_name in self._instances:
                self._instances[self._registry_name].clear()

    def __contains__(self, name: str) -> bool:
        return self.is_registered(name)

    def __len__(self) -> int:
        with self._lock:
            return len(self._instances.get(self._registry_name, {}))

    def __repr__(self) -> str:
        return f"ComponentRegistry(name={self._registry_name!r}, components={self.list_registered()})"


detector_registry: ComponentRegistry = ComponentRegistry("detectors")
tracker_registry: ComponentRegistry = ComponentRegistry("trackers")
optimizer_registry: ComponentRegistry = ComponentRegistry("optimizers")
capture_registry: ComponentRegistry = ComponentRegistry("captures")
exporter_registry: ComponentRegistry = ComponentRegistry("exporters")
