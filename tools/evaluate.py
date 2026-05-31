from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


@click.command()
@click.option("--model-onnx", required=True, help="ONNX model path to evaluate")
@click.option("--model-original", default=None, help="Original model for comparison")
@click.option("--samples", "-n", default=50, help="Number of validation samples")
@click.option("--tolerance", default=0.01, help="Output tolerance threshold")
def evaluate(model_onnx: str, model_original: str | None, samples: int, tolerance: float) -> None:
    from src.utils.logger import setup_logging
    setup_logging(level="INFO", log_format="console")

    console.print(f"\n[bold cyan]ObjaTrack-XL Model Evaluation[/bold cyan]")
    console.print("=" * 50)

    from src.optimization.model_validator import ModelValidator
    validator = ModelValidator(tolerance=tolerance, sample_count=samples)

    console.print(f"\n[yellow]Validating ONNX structure...[/yellow]")
    structure = validator.validate_onnx(model_onnx)

    if structure.get("valid"):
        console.print(f"  [green]✓ Valid ONNX model[/green]")
        console.print(f"  Nodes: {structure.get('num_nodes')}")
        console.print(f"  Opset: {structure.get('opset')}")
        console.print(f"  Size: {structure.get('size_mb')} MB")
        console.print(f"  Op Types: {len(structure.get('op_types', []))}")
    else:
        console.print(f"  [red]✗ Invalid: {structure.get('error')}[/red]")
        return

    if model_original and Path(model_original).exists():
        console.print(f"\n[yellow]Comparing outputs ({samples} samples)...[/yellow]")
        comparison = validator.compare_outputs(model_original, model_onnx)

        if comparison.get("valid"):
            console.print(f"  [green]✓ Within tolerance[/green]")
        else:
            console.print(f"  [red]✗ Exceeds tolerance[/red]")

        console.print(f"  Avg Diff: {comparison.get('avg_absolute_diff', 'N/A')}")
        console.print(f"  Max Diff: {comparison.get('max_absolute_diff', 'N/A')}")
        console.print(f"  Cosine Sim: {comparison.get('avg_cosine_similarity', 'N/A')}")

    console.print(f"\n[yellow]Profiling model...[/yellow]")
    from src.optimization.profiler import ModelProfiler
    profiler = ModelProfiler(model_onnx)
    profile = profiler.profile_onnx()

    console.print(f"  Total Params: {profile.get('total_params_millions', 'N/A')}M")
    console.print(f"  Total Nodes: {profile.get('total_nodes', 'N/A')}")

    console.print(f"\n[yellow]Running inference benchmark...[/yellow]")
    bench = validator.benchmark_inference(model_onnx, iterations=100, warmup=20)
    console.print(f"  Avg Latency: {bench['avg_latency_ms']}ms")
    console.print(f"  Avg FPS: {bench['avg_fps']}")
    console.print(f"  P95: {bench['p95_latency_ms']}ms")

    console.print(f"\n[bold green]Evaluation complete![/bold green]")


if __name__ == "__main__":
    evaluate()
