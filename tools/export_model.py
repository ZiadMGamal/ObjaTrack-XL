from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


@click.command()
@click.option("--model", "-m", default="models/yolov8n.pt", help="Model path")
@click.option("--format", "-f", "export_format", default="onnx", help="Export format")
@click.option("--output-dir", "-o", default="models/optimized", help="Output directory")
@click.option("--opset", default=17, help="ONNX opset version")
@click.option("--imgsz", default=640, help="Image size")
@click.option("--half", is_flag=True, help="FP16 mode")
@click.option("--dynamic", is_flag=True, default=True, help="Dynamic batch")
def export_model(
    model: str,
    export_format: str,
    output_dir: str,
    opset: int,
    imgsz: int,
    half: bool,
    dynamic: bool,
) -> None:
    from src.utils.logger import setup_logging
    setup_logging(level="INFO", log_format="console")

    console.print(f"\n[bold cyan]Exporting model to {export_format.upper()}[/bold cyan]")

    from ultralytics import YOLO
    yolo = YOLO(model)

    result = yolo.export(
        format=export_format,
        opset=opset,
        imgsz=imgsz,
        half=half,
        dynamic=dynamic,
    )

    console.print(f"[green]✓ Export complete: {result}[/green]")


if __name__ == "__main__":
    export_model()
