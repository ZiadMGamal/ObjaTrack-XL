from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option("--config", "-c", default="configs/benchmark.yaml", help="Benchmark config path")
@click.option("--model-pt", default="models/yolov8n.pt", help="PyTorch model path")
@click.option("--model-onnx", default=None, help="ONNX model path")
@click.option("--iterations", "-n", default=200, help="Benchmark iterations")
@click.option("--warmup", "-w", default=50, help="Warmup iterations")
@click.option("--input-size", default=640, help="Input size")
@click.option("--output", "-o", default="outputs/reports", help="Output directory")
def benchmark(
    config: str,
    model_pt: str,
    model_onnx: str | None,
    iterations: int,
    warmup: int,
    input_size: int,
    output: str,
) -> None:
    setup_logging(level="INFO", log_format="console")

    console.print("\n[bold cyan]ObjaTrack-XL Performance Benchmark[/bold cyan]")
    console.print("=" * 50)

    results = {}
    dummy = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

    if Path(model_pt).exists():
        console.print(f"\n[yellow]Benchmarking PyTorch model: {model_pt}[/yellow]")
        results["pytorch"] = _benchmark_pytorch(model_pt, dummy, iterations, warmup, input_size)

    if model_onnx and Path(model_onnx).exists():
        console.print(f"\n[yellow]Benchmarking ONNX model: {model_onnx}[/yellow]")
        results["onnx_cpu"] = _benchmark_onnx(model_onnx, iterations, warmup, input_size, ["CPUExecutionProvider"])

        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                console.print(f"\n[yellow]Benchmarking ONNX model (CUDA): {model_onnx}[/yellow]")
                results["onnx_gpu"] = _benchmark_onnx(
                    model_onnx, iterations, warmup, input_size, ["CUDAExecutionProvider"]
                )
        except ImportError:
            pass

    onnx_int8 = model_onnx.replace(".onnx", "_int8.onnx") if model_onnx else None
    if onnx_int8 and Path(onnx_int8).exists():
        console.print(f"\n[yellow]Benchmarking Quantized model: {onnx_int8}[/yellow]")
        results["onnx_int8"] = _benchmark_onnx(onnx_int8, iterations, warmup, input_size, ["CPUExecutionProvider"])

    if results:
        _display_results(results)
        _save_results(results, output)
        _generate_report(results, output)


def _benchmark_pytorch(
    model_path: str,
    dummy_frame: np.ndarray,
    iterations: int,
    warmup: int,
    input_size: int,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console) as progress:
        task = progress.add_task("Warmup...", total=warmup)
        for _ in range(warmup):
            model.predict(source=dummy_frame, imgsz=input_size, verbose=False)
            progress.advance(task)

    latencies = []
    preprocess_times = []
    postprocess_times = []

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console) as progress:
        task = progress.add_task("Benchmarking...", total=iterations)
        for _ in range(iterations):
            t_start = time.perf_counter()
            results = model.predict(source=dummy_frame, imgsz=input_size, verbose=False)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000)

            if hasattr(results[0], "speed"):
                speed = results[0].speed
                preprocess_times.append(speed.get("preprocess", 0))
                postprocess_times.append(speed.get("postprocess", 0))

            progress.advance(task)

    lat = np.array(latencies)
    return {
        "model": Path(model_path).name,
        "backend": "PyTorch",
        "iterations": iterations,
        "avg_latency_ms": round(float(np.mean(lat)), 2),
        "std_latency_ms": round(float(np.std(lat)), 2),
        "min_latency_ms": round(float(np.min(lat)), 2),
        "max_latency_ms": round(float(np.max(lat)), 2),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p90_ms": round(float(np.percentile(lat, 90)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "avg_fps": round(1000.0 / float(np.mean(lat)), 1),
        "avg_preprocess_ms": round(float(np.mean(preprocess_times)), 2) if preprocess_times else 0,
        "avg_postprocess_ms": round(float(np.mean(postprocess_times)), 2) if postprocess_times else 0,
        "model_size_mb": round(Path(model_path).stat().st_size / (1024 * 1024), 2),
    }


def _benchmark_onnx(
    model_path: str,
    iterations: int,
    warmup: int,
    input_size: int,
    providers: list[str],
) -> dict:
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

    input_info = session.get_inputs()[0]
    shape = [s if isinstance(s, int) else 1 for s in input_info.shape]
    dummy = np.random.randn(*shape).astype(np.float32)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console) as progress:
        task = progress.add_task("Warmup...", total=warmup)
        for _ in range(warmup):
            session.run(None, {input_info.name: dummy})
            progress.advance(task)

    latencies = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console) as progress:
        task = progress.add_task("Benchmarking...", total=iterations)
        for _ in range(iterations):
            start = time.perf_counter()
            session.run(None, {input_info.name: dummy})
            latencies.append((time.perf_counter() - start) * 1000)
            progress.advance(task)

    lat = np.array(latencies)
    active_providers = session.get_providers()

    return {
        "model": Path(model_path).name,
        "backend": f"ONNX ({active_providers[0].replace('ExecutionProvider', '')})",
        "iterations": iterations,
        "avg_latency_ms": round(float(np.mean(lat)), 2),
        "std_latency_ms": round(float(np.std(lat)), 2),
        "min_latency_ms": round(float(np.min(lat)), 2),
        "max_latency_ms": round(float(np.max(lat)), 2),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p90_ms": round(float(np.percentile(lat, 90)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "avg_fps": round(1000.0 / float(np.mean(lat)), 1),
        "avg_preprocess_ms": 0,
        "avg_postprocess_ms": 0,
        "model_size_mb": round(Path(model_path).stat().st_size / (1024 * 1024), 2),
    }


def _display_results(results: dict) -> None:
    console.print("\n[bold cyan]Benchmark Results[/bold cyan]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    for name in results:
        table.add_column(name.upper(), justify="right")

    metrics = [
        ("Backend", "backend"),
        ("Model Size (MB)", "model_size_mb"),
        ("Avg Latency (ms)", "avg_latency_ms"),
        ("Std Dev (ms)", "std_latency_ms"),
        ("Min Latency (ms)", "min_latency_ms"),
        ("P50 (ms)", "p50_ms"),
        ("P90 (ms)", "p90_ms"),
        ("P95 (ms)", "p95_ms"),
        ("P99 (ms)", "p99_ms"),
        ("Avg FPS", "avg_fps"),
        ("Preprocess (ms)", "avg_preprocess_ms"),
        ("Postprocess (ms)", "avg_postprocess_ms"),
    ]

    for label, key in metrics:
        row = [label]
        for name in results:
            val = results[name].get(key, "N/A")
            row.append(str(val))
        table.add_row(*row)

    console.print(table)

    if len(results) > 1:
        console.print("\n[bold cyan]Speedup Analysis[/bold cyan]")
        names = list(results.keys())
        base = results[names[0]]["avg_latency_ms"]
        for name in names[1:]:
            other = results[name]["avg_latency_ms"]
            speedup = base / other if other > 0 else 0
            console.print(f"  {name} vs {names[0]}: [green]{speedup:.2f}x[/green] speedup")


def _save_results(results: dict, output_dir: str) -> None:
    import json
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "benchmark_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\nResults saved to: {path}")


def _generate_report(results: dict, output_dir: str) -> None:
    from src.io.report_generator import ReportGenerator

    report = ReportGenerator(
        output_path=str(Path(output_dir) / "benchmark_report.html"),
        title="ObjaTrack-XL Benchmark Report",
    )

    for name, data in results.items():
        report.add_summary_card(
            f"{name} FPS", str(data["avg_fps"]), "fps",
            "good" if data["avg_fps"] >= 30 else "warn" if data["avg_fps"] >= 15 else "bad",
        )

    headers = ["Metric"] + [n.upper() for n in results]
    rows = []
    for label in ["avg_latency_ms", "p50_ms", "p95_ms", "avg_fps", "model_size_mb"]:
        row = [label]
        for name in results:
            row.append(str(results[name].get(label, "N/A")))
        rows.append(row)

    report.set_benchmark_table(headers, rows)

    from src.utils.device import DeviceManager
    dm = DeviceManager()
    report.set_system_info(dm.get_system_info())

    path = report.generate()
    console.print(f"Report saved to: {path}")


if __name__ == "__main__":
    benchmark()
