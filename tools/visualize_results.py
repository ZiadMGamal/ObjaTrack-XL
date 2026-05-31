from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


@click.command()
@click.option("--results", "-r", required=True, help="Path to JSON results file")
@click.option("--output", "-o", default="outputs/reports", help="Output directory")
@click.option("--format", "-f", "out_format", default="html", help="Output format (html)")
def visualize_results(results: str, output: str, out_format: str) -> None:
    import json
    from src.io.report_generator import ReportGenerator

    with open(results) as f:
        data = json.load(f)

    report = ReportGenerator(
        output_path=str(Path(output) / "visualization_report.html"),
        title="ObjaTrack-XL Results Visualization",
    )

    if isinstance(data, dict) and "frames" in data:
        total_frames = len(data["frames"])
        total_detections = sum(len(f.get("detections", [])) for f in data["frames"])
        total_tracks = sum(len(f.get("tracks", [])) for f in data["frames"])

        report.add_summary_card("Total Frames", str(total_frames), "frames")
        report.add_summary_card("Total Detections", str(total_detections), "detections")
        report.add_summary_card("Total Tracks", str(total_tracks), "tracks")
        report.add_summary_card(
            "Avg Det/Frame",
            str(round(total_detections / max(1, total_frames), 1)),
            "avg",
        )

    path = report.generate()
    console.print(f"[green]✓ Report generated: {path}[/green]")


if __name__ == "__main__":
    visualize_results()
