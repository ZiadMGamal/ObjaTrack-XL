from __future__ import annotations

import numpy as np
import pytest

from src.visualization.color_palette import ColorPalette
from src.io.json_exporter import JSONExporter
from src.io.csv_exporter import CSVExporter


class TestColorPalette:

    def test_get_color(self) -> None:
        palette = ColorPalette()
        color = palette.get_color(0)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    def test_color_consistency(self) -> None:
        palette = ColorPalette()
        c1 = palette.get_color(5)
        c2 = palette.get_color(5)
        assert c1 == c2

    def test_different_ids_different_colors(self) -> None:
        palette = ColorPalette()
        colors = {palette.get_color(i) for i in range(20)}
        assert len(colors) >= 15

    def test_class_color(self) -> None:
        palette = ColorPalette()
        color = palette.get_class_color("person")
        assert len(color) == 3

    def test_contrast_color(self) -> None:
        palette = ColorPalette()
        white = palette.get_contrast_color((255, 255, 255))
        assert white == (0, 0, 0)
        black = palette.get_contrast_color((0, 0, 0))
        assert black == (255, 255, 255)

    def test_gradient_color(self) -> None:
        palette = ColorPalette()
        color = palette.get_gradient_color(0.5)
        assert len(color) == 3


class TestJSONExporter:

    def test_initialization(self, tmp_path) -> None:
        path = str(tmp_path / "test.json")
        exporter = JSONExporter(output_path=path)
        assert exporter.output_path.name == "test.json"

    def test_add_frame(self, tmp_path) -> None:
        path = str(tmp_path / "test.json")
        exporter = JSONExporter(output_path=path)
        exporter.add_frame_result(
            frame_id=1,
            timestamp=1000.0,
            detections=[{"box": [10, 10, 50, 50], "score": 0.9}],
        )
        result = exporter.export()
        assert result == path

    def test_clear(self, tmp_path) -> None:
        path = str(tmp_path / "test.json")
        exporter = JSONExporter(output_path=path)
        exporter.add_frame_result(1, 1000.0, [])
        exporter.clear()


class TestCSVExporter:

    def test_initialization(self, tmp_path) -> None:
        path = str(tmp_path / "test.csv")
        exporter = CSVExporter(output_path=path, mode="tracks")
        assert exporter.output_path.name == "test.csv"

    def test_add_tracks(self, tmp_path) -> None:
        path = str(tmp_path / "test.csv")
        exporter = CSVExporter(output_path=path, mode="tracks")
        exporter.add_tracks(
            frame_id=1,
            timestamp=1000.0,
            tracks=[{
                "track_id": 1,
                "box": [10, 10, 50, 50],
                "score": 0.9,
                "class_id": 0,
                "class_name": "person",
                "age": 5,
                "hits": 5,
                "speed": 3.5,
                "direction": 45.0,
            }],
        )
        result = exporter.export()
        assert result == path
