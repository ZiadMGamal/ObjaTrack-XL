from __future__ import annotations

import sys
from pathlib import Path

import click

from src.config.settings import Settings
from src.core.engine import PipelineEngine
from src.utils.logger import setup_logging, get_logger


@click.group()
@click.version_option(version="1.0.0", prog_name="ObjaTrack-XL")
def cli() -> None:
    pass


@cli.command()
@click.option("--config", "-c", default="configs/default.yaml", help="Path to configuration file")
@click.option("--source", "-s", default=None, help="Video source (file path, webcam index, or RTSP URL)")
@click.option("--model", "-m", default=None, help="Model name or path")
@click.option("--tracker", "-t", default=None, help="Tracker type (sort, bytetrack, botsort, deepsort)")
@click.option("--confidence", default=None, type=float, help="Confidence threshold")
@click.option("--device", "-d", default=None, help="Device (auto, cpu, cuda)")
@click.option("--save-video", is_flag=True, default=False, help="Save output video")
@click.option("--save-json", is_flag=True, default=False, help="Save JSON results")
@click.option("--no-display", is_flag=True, default=False, help="Run without display")
@click.option("--log-level", default=None, help="Logging level")
def run(
    config: str,
    source: str | None,
    model: str | None,
    tracker: str | None,
    confidence: float | None,
    device: str | None,
    save_video: bool,
    save_json: bool,
    no_display: bool,
    log_level: str | None,
) -> None:
    config_path = Path(config)
    if config_path.exists():
        settings = Settings.from_yaml(config)
    else:
        settings = Settings()

    if source:
        if source.isdigit():
            settings.source.type = "webcam"
            settings.source.webcam_index = int(source)
        elif source.startswith("rtsp://"):
            settings.source.type = "rtsp"
            settings.source.rtsp_url = source
        else:
            settings.source.type = "file"
            settings.source.path = source

    if model:
        settings.model.name = model
    if tracker:
        settings.tracker.type = tracker
    if confidence is not None:
        settings.model.confidence_threshold = confidence
    if device:
        settings.device = device
    if save_video:
        settings.output.save_video = True
    if save_json:
        settings.output.save_json = True
    if log_level:
        settings.logging.level = log_level

    setup_logging(level=settings.logging.level, log_format=settings.logging.format)
    log = get_logger("main")
    log.info("starting_objatrack_xl", version="1.0.0")

    engine = PipelineEngine(settings)
    engine.initialize()
    engine.run()


@cli.command()
@click.option("--config", "-c", default="configs/default.yaml", help="Path to configuration file")
def validate(config: str) -> None:
    from src.config.validator import ConfigValidator

    config_path = Path(config)
    if config_path.exists():
        settings = Settings.from_yaml(config)
    else:
        settings = Settings()

    validator = ConfigValidator(settings)
    valid = validator.validate()

    report = validator.get_report()

    click.echo(f"\nConfiguration: {'VALID' if valid else 'INVALID'}")
    click.echo(f"Config file: {config}")

    if report["errors"]:
        click.echo(f"\nErrors ({len(report['errors'])}):")
        for e in report["errors"]:
            click.echo(f"  ✗ {e}")

    if report["warnings"]:
        click.echo(f"\nWarnings ({len(report['warnings'])}):")
        for w in report["warnings"]:
            click.echo(f"  ⚠ {w}")

    click.echo(f"\nSettings:")
    for k, v in report["settings_summary"].items():
        click.echo(f"  {k}: {v}")


@cli.command()
def info() -> None:
    from src.utils.device import DeviceManager

    dm = DeviceManager()
    info = dm.get_system_info()

    click.echo("\nObjaTrack-XL System Information")
    click.echo("=" * 40)
    for k, v in info.items():
        click.echo(f"  {k}: {v}")

    click.echo(f"\nAvailable Detectors: yolo, onnx, tensorrt")
    click.echo(f"Available Trackers: sort, bytetrack, botsort, deepsort")


@cli.command()
def models() -> None:
    from src.utils.download import ModelDownloader

    downloader = ModelDownloader()
    model_list = downloader.list_models()

    if not model_list:
        click.echo("No models found in models/ directory")
        return

    click.echo(f"\nInstalled Models ({len(model_list)}):")
    click.echo("-" * 50)
    for m in model_list:
        click.echo(f"  {m['name']:30s} {m['size_mb']:8.2f} MB  [{m['format']}]")


if __name__ == "__main__":
    cli()
