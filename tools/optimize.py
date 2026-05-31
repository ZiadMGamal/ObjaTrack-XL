from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger

console = Console()


@click.command()
@click.option("--model", "-m", default="models/yolov8n.pt", help="Source model path")
@click.option("--output-dir", "-o", default="models/optimized", help="Output directory")
@click.option("--format", "-f", "export_format", default="onnx", help="Export format (onnx, tensorrt)")
@click.option("--opset", default=17, help="ONNX opset version")
@click.option("--dynamic/--no-dynamic", default=True, help="Dynamic axes")
@click.option("--simplify/--no-simplify", default=True, help="Simplify ONNX model")
@click.option("--half", is_flag=True, default=False, help="FP16 export")
@click.option("--quantize", "-q", is_flag=True, default=False, help="Apply INT8 quantization")
@click.option("--quantize-method", default="dynamic", help="Quantization method (dynamic, static)")
@click.option("--validate/--no-validate", default=True, help="Validate after export")
@click.option("--benchmark", "-b", is_flag=True, default=False, help="Run benchmark after export")
def optimize(
    model: str,
    output_dir: str,
    export_format: str,
    opset: int,
    dynamic: bool,
    simplify: bool,
    half: bool,
    quantize: bool,
    quantize_method: str,
    validate: bool,
    benchmark: bool,
) -> None:
    setup_logging(level="INFO", log_format="console")

    console.print("\n[bold cyan]ObjaTrack-XL Model Optimization Pipeline[/bold cyan]")
    console.print("=" * 50)

    if not Path(model).exists():
        console.print(f"[red]Model not found: {model}[/red]")
        console.print("[yellow]Downloading model...[/yellow]")
        from src.utils.download import ModelDownloader
        downloader = ModelDownloader()
        model = str(downloader.download(Path(model).name))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    onnx_path = None
    if export_format == "onnx":
        console.print(f"\n[yellow]Step 1: Exporting to ONNX[/yellow]")
        console.print(f"  Model: {model}")
        console.print(f"  Opset: {opset}")
        console.print(f"  Dynamic: {dynamic}")
        console.print(f"  Simplify: {simplify}")
        console.print(f"  Half: {half}")

        from src.optimization.onnx_exporter import ONNXExporter
        exporter = ONNXExporter(
            model_path=model,
            output_dir=output_dir,
            opset_version=opset,
            dynamic_axes=dynamic,
            simplify=simplify,
            half=half,
        )
        exporter.initialize()
        onnx_path = exporter.optimize()

        results = exporter.optimization_results
        console.print(f"\n  [green]✓ Export complete[/green]")
        console.print(f"  Output: {onnx_path}")
        console.print(f"  Original: {results.get('original_size_mb', 'N/A')} MB")
        console.print(f"  Optimized: {results.get('optimized_size_mb', 'N/A')} MB")
        console.print(f"  Reduction: {results.get('reduction_percent', 'N/A')}%")

    elif export_format == "tensorrt":
        console.print(f"\n[yellow]Step 1: Exporting to TensorRT[/yellow]")
        from src.optimization.trt_exporter import TensorRTExporter
        trt_exporter = TensorRTExporter(
            model_path=model,
            output_dir=output_dir,
            fp16_mode=half,
        )
        engine_path = trt_exporter.export()
        console.print(f"  [green]✓ TensorRT export complete: {engine_path}[/green]")

    if quantize and onnx_path:
        console.print(f"\n[yellow]Step 2: Quantization ({quantize_method})[/yellow]")

        from src.optimization.quantizer import ModelQuantizer
        quantizer = ModelQuantizer(
            model_path=onnx_path,
            output_dir=output_dir,
            method=quantize_method,
            precision="int8",
        )
        quantizer.initialize()
        quant_path = quantizer.optimize()

        quant_results = quantizer.optimization_results
        console.print(f"  [green]✓ Quantization complete[/green]")
        console.print(f"  Output: {quant_path}")
        console.print(f"  Reduction: {quant_results.get('reduction_percent', 'N/A')}%")

    if validate and onnx_path:
        console.print(f"\n[yellow]Step 3: Validation[/yellow]")

        from src.optimization.model_validator import ModelValidator
        validator = ModelValidator()
        val_result = validator.validate_onnx(onnx_path)

        if val_result.get("valid"):
            console.print(f"  [green]✓ Model is valid[/green]")
            console.print(f"  Nodes: {val_result.get('num_nodes', 'N/A')}")
            console.print(f"  Opset: {val_result.get('opset', 'N/A')}")
        else:
            console.print(f"  [red]✗ Validation failed: {val_result.get('error', 'Unknown')}[/red]")

    if benchmark and onnx_path:
        console.print(f"\n[yellow]Step 4: Quick Benchmark[/yellow]")

        from src.optimization.model_validator import ModelValidator
        validator = ModelValidator()
        bench = validator.benchmark_inference(onnx_path, iterations=100, warmup=20)
        console.print(f"  Avg Latency: {bench['avg_latency_ms']}ms")
        console.print(f"  Avg FPS: {bench['avg_fps']}")
        console.print(f"  P95: {bench['p95_latency_ms']}ms")

    console.print(f"\n[bold green]Optimization pipeline complete![/bold green]")


if __name__ == "__main__":
    optimize()
