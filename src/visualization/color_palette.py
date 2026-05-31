from __future__ import annotations

import colorsys


class ColorPalette:
    _PRESET_COLORS = [
        (255, 64, 64),
        (64, 255, 64),
        (64, 64, 255),
        (255, 255, 64),
        (255, 64, 255),
        (64, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 255, 128),
        (255, 0, 128),
        (128, 255, 0),
        (0, 128, 255),
        (255, 192, 64),
        (192, 64, 255),
        (64, 255, 192),
        (255, 64, 192),
        (192, 255, 64),
        (64, 192, 255),
        (128, 128, 255),
        (255, 128, 128),
        (128, 255, 128),
        (200, 100, 50),
        (50, 200, 100),
        (100, 50, 200),
        (220, 180, 60),
        (60, 220, 180),
        (180, 60, 220),
        (240, 120, 90),
        (90, 240, 120),
        (120, 90, 240),
        (255, 160, 160),
        (160, 255, 160),
    ]

    def __init__(self, num_colors: int = 256, saturation: float = 0.85, value: float = 0.95) -> None:
        self._num_colors = num_colors
        self._saturation = saturation
        self._value = value
        self._cache: dict[int, tuple[int, int, int]] = {}
        self._class_colors: dict[str, tuple[int, int, int]] = {}

    def get_color(self, track_id: int) -> tuple[int, int, int]:
        if track_id in self._cache:
            return self._cache[track_id]

        if track_id < len(self._PRESET_COLORS):
            color = self._PRESET_COLORS[track_id % len(self._PRESET_COLORS)]
        else:
            hue = (track_id * 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, self._saturation, self._value)
            color = (int(r * 255), int(g * 255), int(b * 255))

        self._cache[track_id] = color
        return color

    def get_class_color(self, class_name: str) -> tuple[int, int, int]:
        if class_name in self._class_colors:
            return self._class_colors[class_name]

        hash_val = hash(class_name)
        color_idx = abs(hash_val) % len(self._PRESET_COLORS)
        color = self._PRESET_COLORS[color_idx]
        self._class_colors[class_name] = color
        return color

    def set_class_color(self, class_name: str, color: tuple[int, int, int]) -> None:
        self._class_colors[class_name] = color

    def get_contrast_color(self, bg_color: tuple[int, int, int]) -> tuple[int, int, int]:
        luminance = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]) / 255
        return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)

    def get_gradient_color(
        self,
        value: float,
        min_val: float = 0.0,
        max_val: float = 1.0,
        start_color: tuple[int, int, int] = (0, 255, 0),
        end_color: tuple[int, int, int] = (255, 0, 0),
    ) -> tuple[int, int, int]:
        t = max(0.0, min(1.0, (value - min_val) / (max_val - min_val) if max_val != min_val else 0.0))
        r = int(start_color[0] + t * (end_color[0] - start_color[0]))
        g = int(start_color[1] + t * (end_color[1] - start_color[1]))
        b = int(start_color[2] + t * (end_color[2] - start_color[2]))
        return (r, g, b)

    @staticmethod
    def bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
        return (color[2], color[1], color[0])

    @staticmethod
    def rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
        return (color[2], color[1], color[0])

    def clear_cache(self) -> None:
        self._cache.clear()
