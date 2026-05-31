<div align="center">

# 🎯 ObjaTrack-XL

### High-Performance Edge-Optimized Real-Time Object Detection & Tracking System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Optimized-FF6F00?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Multi--Arch-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<p align="center">
  <strong>Production-grade computer vision pipeline with multi-backend inference, 4 tracking algorithms, real-time analytics, model optimization toolkit, and REST API — designed for edge deployment.</strong>
</p>

---

**Created by [Ziad Mohamed Gamal](https://github.com/ZiadMGamal)**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Model Optimization](#-model-optimization)
- [Benchmarking](#-benchmarking)
- [Docker Deployment](#-docker-deployment)
- [Testing](#-testing)
- [License](#-license)

---

## 🔍 Overview

**ObjaTrack-XL** is a modular, production-ready real-time object detection and tracking system built with a focus on **edge deployment performance**. The system demonstrates deep understanding of the full computer vision lifecycle — from raw inference through model optimization (ONNX/TensorRT conversion, INT8 quantization) to containerized deployment.

### What Makes This Different

| Aspect | Implementation |
|---|---|
| **Multi-Backend Inference** | PyTorch (Ultralytics), ONNX Runtime, TensorRT — hot-swappable via factory pattern |
| **4 Tracking Algorithms** | SORT, ByteTrack, BoT-SORT, DeepSORT — each with Kalman filtering and configurable association |
| **Full Optimization Pipeline** | ONNX export → graph simplification → INT8/FP16 quantization → validation → benchmarking |
| **Real-Time Analytics** | Line counting, zone occupancy, speed estimation, dwell time, event detection |
| **Production Infrastructure** | REST API, Docker (CPU/GPU), CI/CD, structured logging, comprehensive testing |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        ObjaTrack-XL Engine                       │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│ Capture  │Detection │ Tracking │Analytics │   Viz    │   I/O    │
│          │          │          │          │          │          │
│ Webcam   │ YOLO     │ SORT     │ Counter  │ Renderer │ Video    │
│ File     │ ONNX     │ ByteTrack│ Zones    │ HUD      │ JSON     │
│ RTSP     │ TensorRT │ BoTSORT  │ Speed    │ Heatmap  │ CSV      │
│ Buffer   │ NMS(4x)  │ DeepSORT │ Dwell    │ Trails   │ COCO     │
│          │ Preproc  │ Kalman   │ Events   │ Palette  │ Report   │
├──────────┴──────────┴──────────┴──────────┴──────────┴──────────┤
│                    Core Infrastructure                           │
│  Config (Pydantic) │ Registry Pattern │ Exceptions │ Metrics    │
│  Logging (struct)  │ Base Classes     │ Factories  │ Profiling  │
├──────────────────────────────────────────────────────────────────┤
│  Optimization: ONNX Export → Simplify → Quantize → Validate     │
├──────────────────────────────────────────────────────────────────┤
│  Deployment: CLI (Click) │ REST API (FastAPI) │ Docker (CPU/GPU) │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🔬 Detection Engine
- **Multi-backend support** — YOLOv8 (PyTorch), ONNX Runtime, TensorRT
- **4 NMS algorithms** — Standard, Soft-NMS, Batched (per-class), DIoU-NMS
- **Custom preprocessing** — Letterboxing, normalization, batch support
- **Factory pattern** — Runtime backend selection via configuration

### 🎯 Tracking System
- **SORT** — Simple Online and Realtime Tracking with Kalman filter
- **ByteTrack** — Two-stage association (high/low confidence detections)
- **BoT-SORT** — Score fusion with optional Re-ID integration
- **DeepSORT** — Appearance feature matching with cosine distance
- **Kalman Filters** — XYAH and XYWH state-space variants
- **Association** — Hungarian, greedy, IoU, cosine, Euclidean, Mahalanobis distance metrics

### 📊 Real-Time Analytics
- **Line Counting** — Directional IN/OUT counting with per-class breakdown
- **Zone Analytics** — Polygon-based occupancy monitoring with entry/exit events
- **Speed Estimation** — Pixel-to-meter calibrated velocity tracking
- **Dwell Time** — Threshold-based alerting for lingering objects
- **Event Detection** — Loitering, speed violations, crowd formation, abandoned objects

### 🎨 Visualization
- **Styled Bounding Boxes** — Corner accents, semi-transparent labels
- **Trajectory Trails** — Smoothed, fading paths with direction arrows
- **Detection Heatmap** — Gaussian accumulation with temporal decay
- **HUD Overlay** — Real-time FPS, latency, detection/track counts, system info

### ⚡ Optimization Toolkit
- **ONNX Export** — Dynamic axes, opset control, graph simplification
- **Quantization** — Dynamic & static INT8 quantization with calibration
- **Validation** — Output comparison, cosine similarity, tolerance checking
- **Profiling** — Layer analysis, FLOPS estimation, percentile latency benchmarks
- **TensorRT** — FP16/INT8 engine conversion via Ultralytics

### 🏭 Production Infrastructure
- **REST API** — FastAPI with `/detect`, `/detect-and-track`, `/metrics`, `/health`
- **Docker** — Multi-architecture (CPU/GPU) with health checks
- **CI/CD** — GitHub Actions with lint, multi-OS tests, Docker build, security scan
- **Structured Logging** — JSON/console output via structlog
- **Pydantic Config** — Strict validation, YAML parsing, environment overrides

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.10+ (strict typing, OOP, no inline comments) |
| **Detection** | Ultralytics YOLOv8, ONNX Runtime, TensorRT |
| **Tracking** | Custom SORT/ByteTrack/BoTSORT/DeepSORT with SciPy |
| **Optimization** | ONNX, onnxsim, onnxruntime.quantization |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Config** | Pydantic Settings, PyYAML, Click CLI |
| **Metrics** | psutil, GPUtil, custom profilers |
| **Visualization** | OpenCV, NumPy |
| **Logging** | structlog (JSON + console) |
| **Testing** | pytest, pytest-cov, pytest-xdist |
| **CI/CD** | GitHub Actions, Docker, Ruff, mypy, Bandit |
| **Containerization** | Docker (CPU slim + CUDA 12.2 GPU) |

---

## 📁 Project Structure

```
ObjaTrack-XL/
├── main.py                          # CLI entry point (Click)
├── pyproject.toml                   # Project metadata & tool config
├── requirements.txt                 # Dependencies
├── Dockerfile                       # CPU container
├── Dockerfile.gpu                   # GPU container (CUDA 12.2)
├── docker-compose.yml               # Multi-service orchestration
├── Makefile                         # Build automation
├── configs/
│   ├── default.yaml                 # Default pipeline config
│   └── benchmark.yaml               # Benchmark config
├── src/
│   ├── core/
│   │   ├── base.py                  # Abstract interfaces & data classes
│   │   ├── engine.py                # Main pipeline orchestrator
│   │   ├── exceptions.py            # Exception hierarchy
│   │   └── registry.py              # Component registry pattern
│   ├── config/
│   │   ├── settings.py              # Pydantic settings models
│   │   └── validator.py             # Configuration validator
│   ├── capture/
│   │   ├── base_capture.py          # Abstract capture interface
│   │   ├── webcam_capture.py        # Webcam source
│   │   ├── file_capture.py          # Video file source
│   │   ├── rtsp_capture.py          # RTSP stream source
│   │   ├── frame_buffer.py          # Thread-safe frame buffer
│   │   └── stream_manager.py        # Multi-source manager
│   ├── detection/
│   │   ├── base_detector.py         # Abstract detector
│   │   ├── yolo_detector.py         # YOLOv8 (Ultralytics)
│   │   ├── onnx_detector.py         # ONNX Runtime inference
│   │   ├── trt_detector.py          # TensorRT inference
│   │   ├── detector_factory.py      # Factory pattern
│   │   ├── nms.py                   # NMS algorithms (4 variants)
│   │   └── preprocessing.py         # Input preprocessing pipeline
│   ├── tracking/
│   │   ├── track.py                 # Track data class
│   │   ├── base_tracker.py          # Abstract tracker
│   │   ├── sort_tracker.py          # SORT algorithm
│   │   ├── byte_tracker.py          # ByteTrack algorithm
│   │   ├── bot_sort_tracker.py      # BoT-SORT algorithm
│   │   ├── deep_sort_tracker.py     # DeepSORT algorithm
│   │   ├── tracker_factory.py       # Factory pattern
│   │   ├── kalman_filter.py         # Kalman filter (XYAH + XYWH)
│   │   └── association.py           # Association algorithms
│   ├── visualization/
│   │   ├── renderer.py              # Overlay rendering
│   │   ├── hud.py                   # Heads-up display
│   │   ├── trajectory.py            # Trajectory visualization
│   │   ├── heatmap.py               # Detection heatmap
│   │   └── color_palette.py         # Color management
│   ├── analytics/
│   │   ├── counter.py               # Line crossing counter
│   │   ├── zone.py                  # Zone occupancy analytics
│   │   ├── speed_estimator.py       # Speed estimation
│   │   ├── dwell_time.py            # Dwell time analysis
│   │   └── event_detector.py        # Event detection system
│   ├── metrics/
│   │   ├── fps_counter.py           # FPS measurement
│   │   ├── latency_tracker.py       # Pipeline latency profiling
│   │   ├── memory_monitor.py        # CPU/GPU memory tracking
│   │   ├── system_monitor.py        # System resource monitoring
│   │   └── metrics_aggregator.py    # Central metrics hub
│   ├── optimization/
│   │   ├── base_optimizer.py        # Abstract optimizer
│   │   ├── onnx_exporter.py         # ONNX export + simplification
│   │   ├── quantizer.py             # INT8 quantization
│   │   ├── model_validator.py       # Output validation
│   │   ├── profiler.py              # Model profiling
│   │   └── trt_exporter.py          # TensorRT export
│   ├── io/
│   │   ├── video_writer.py          # Video output
│   │   ├── json_exporter.py         # JSON results export
│   │   ├── csv_exporter.py          # CSV results export
│   │   ├── coco_exporter.py         # COCO format export
│   │   └── report_generator.py      # HTML report generation
│   ├── api/
│   │   └── app.py                   # FastAPI REST API
│   └── utils/
│       ├── logger.py                # Structured logging
│       ├── timer.py                 # Performance timing
│       ├── device.py                # Device detection
│       ├── download.py              # Model downloader
│       ├── image_utils.py           # Image processing
│       └── math_utils.py            # Mathematical utilities
├── tools/
│   ├── benchmark.py                 # Performance benchmarking
│   ├── optimize.py                  # Optimization pipeline
│   ├── export_model.py              # Model export utility
│   ├── evaluate.py                  # Model evaluation
│   ├── visualize_results.py         # Results visualization
│   └── stream_test.py              # Stream connectivity test
├── tests/
│   ├── conftest.py                  # Test fixtures
│   └── unit/
│       ├── test_nms.py              # NMS algorithm tests
│       ├── test_track.py            # Track data class tests
│       ├── test_kalman_filter.py    # Kalman filter tests
│       ├── test_association.py      # Association algorithm tests
│       ├── test_analytics.py        # Analytics module tests
│       ├── test_metrics.py          # Metrics module tests
│       ├── test_preprocessing.py    # Preprocessing tests
│       ├── test_utils.py            # Utility tests
│       └── test_io.py               # I/O module tests
└── .github/
    └── workflows/
        └── ci.yml                   # CI/CD pipeline
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Standard Installation

```bash
git clone https://github.com/ZiadMGamal/ObjaTrack-XL.git
cd ObjaTrack-XL

python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### GPU Support

```bash
pip install onnxruntime-gpu GPUtil
```

### Development Setup

```bash
pip install -e ".[dev]"
```

---

## ⚡ Quick Start

### Run with Video File

```bash
python main.py run --source data/videos/sample.mp4
```

### Run with Webcam

```bash
python main.py run --source 0
```

### Run with RTSP Stream

```bash
python main.py run --source "rtsp://username:password@192.168.1.100:554/stream"
```

### Run with Custom Options

```bash
python main.py run \
    --source data/videos/traffic.mp4 \
    --model yolov8s.pt \
    --tracker bytetrack \
    --confidence 0.3 \
    --save-video \
    --save-json
```

### System Information

```bash
python main.py info
```

### Validate Configuration

```bash
python main.py validate --config configs/default.yaml
```

---

## ⚙️ Configuration

All settings are managed via YAML configuration with Pydantic validation:

```yaml
project_name: "ObjaTrack-XL"
device: "auto"

source:
  type: "file"
  path: "data/videos/sample.mp4"

model:
  name: "yolov8n.pt"
  confidence_threshold: 0.25
  iou_threshold: 0.45
  input_size: [640, 640]
  half_precision: false

tracker:
  type: "bytetrack"       # sort | bytetrack | botsort | deepsort
  max_age: 30
  min_hits: 3

analytics:
  counting:
    enabled: true
    line_start: [0, 360]
    line_end: [1280, 360]
  speed_estimation:
    enabled: true
    pixels_per_meter: 8.0

output:
  save_video: true
  save_json: true
```

Environment variable overrides: `OBJATRACK_DEVICE=cuda`, `OBJATRACK_LOG_LEVEL=DEBUG`

---

## 🌐 API Reference

### Start API Server

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `GET` | `/status` | System status |
| `POST` | `/detect` | Single image detection |
| `POST` | `/detect-and-track` | Detection + tracking |
| `GET` | `/metrics` | Performance metrics |
| `POST` | `/reset-tracker` | Reset tracker state |
| `GET` | `/info` | System information |

### Example Request

```bash
curl -X POST "http://localhost:8000/detect" \
     -F "file=@image.jpg" \
     -F "confidence=0.3"
```

---

## ⚡ Model Optimization

### Full Optimization Pipeline

```bash
python tools/optimize.py \
    --model models/yolov8n.pt \
    --format onnx \
    --opset 17 \
    --simplify \
    --quantize \
    --validate \
    --benchmark
```

### Export Only

```bash
python tools/export_model.py --model models/yolov8n.pt --format onnx --half
```

### Evaluate Model

```bash
python tools/evaluate.py \
    --model-onnx models/optimized/yolov8n.onnx \
    --model-original models/optimized/yolov8n_original.onnx
```

---

## 📊 Benchmarking

### Run Benchmark

```bash
python tools/benchmark.py \
    --model-pt models/yolov8n.pt \
    --model-onnx models/optimized/yolov8n.onnx \
    --iterations 200
```

### Output Includes
- Average / P50 / P90 / P95 / P99 latency
- FPS comparison across backends
- Speedup analysis
- HTML report with visualizations

---

## 🐳 Docker Deployment

### CPU

```bash
docker build -t objatrack-xl:cpu -f Dockerfile .
docker run --rm -v $(pwd)/models:/app/models objatrack-xl:cpu
```

### GPU

```bash
docker build -t objatrack-xl:gpu -f Dockerfile.gpu .
docker run --rm --gpus all -v $(pwd)/models:/app/models objatrack-xl:gpu
```

### Docker Compose

```bash
docker-compose up objatrack-api
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific module
pytest tests/unit/test_nms.py -v
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
<strong>ObjaTrack-XL</strong> — Built for performance. Designed for production.
<br>
<sub>Created by <a href="https://github.com/ZiadMGamal">Ziad Mohamed Gamal</a></sub>
</div>
