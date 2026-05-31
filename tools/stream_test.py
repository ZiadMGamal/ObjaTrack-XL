from __future__ import annotations

import sys
from pathlib import Path

import click
import cv2
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


@click.command()
@click.option("--source", "-s", required=True, help="Video source (file, webcam index, RTSP URL)")
@click.option("--duration", "-d", default=10, help="Test duration in seconds")
def stream_test(source: str, duration: int) -> None:
    import time

    console.print(f"\n[bold cyan]Stream Connectivity Test[/bold cyan]")
    console.print(f"Source: {source}")
    console.print(f"Duration: {duration}s")
    console.print("=" * 40)

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        source_type = "webcam"
    elif source.startswith("rtsp://"):
        cap = cv2.VideoCapture(source)
        source_type = "rtsp"
    else:
        cap = cv2.VideoCapture(source)
        source_type = "file"

    if not cap.isOpened():
        console.print(f"[red]✗ Failed to open source: {source}[/red]")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    console.print(f"[green]✓ Connected![/green]")
    console.print(f"  Type: {source_type}")
    console.print(f"  Resolution: {width}x{height}")
    console.print(f"  FPS: {fps}")

    frames = 0
    errors = 0
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if ret:
            frames += 1
        else:
            errors += 1

    elapsed = time.time() - start
    actual_fps = frames / elapsed if elapsed > 0 else 0

    cap.release()

    console.print(f"\n[cyan]Results:[/cyan]")
    console.print(f"  Frames read: {frames}")
    console.print(f"  Errors: {errors}")
    console.print(f"  Actual FPS: {actual_fps:.1f}")
    console.print(f"  Drop rate: {errors / max(1, frames + errors) * 100:.1f}%")
    console.print(f"  Duration: {elapsed:.1f}s")

    status = "PASS" if errors == 0 and frames > 0 else "WARN" if frames > 0 else "FAIL"
    color = "green" if status == "PASS" else "yellow" if status == "WARN" else "red"
    console.print(f"\n  Status: [{color}]{status}[/{color}]")


if __name__ == "__main__":
    stream_test()
