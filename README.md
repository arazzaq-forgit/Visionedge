# VisionEdge

**Hardware-Accelerated Video Pipeline** — a real-time computer vision system built with Python, TensorRT, CuPy, React, and WebRTC.

Built as part of the Axlero Solutions Intern Project Program (Project 1, 2026 cycle).

---

## Overview

VisionEdge tackles a real bottleneck in production computer vision: running object detection on video streams without the CPU↔GPU data-copy overhead that kills throughput in naive pipelines. The system decodes video, runs inference, and streams annotated results to a browser — with the compute-heavy path kept entirely on the GPU wherever possible.

**Problem it solves:** standard OpenCV + PyTorch pipelines bottleneck on CPU decode and CPU↔GPU tensor transfers, making it hard to process multiple high-resolution streams in real time. VisionEdge addresses this with hardware-accelerated decoding, TensorRT-optimized inference, and a CuPy-based zero-copy pipeline.

---

## Architecture

```
Video Source → Hardware Decode → TensorRT Inference → Zero-Copy Pipeline → WebRTC Stream → React Dashboard
                  (PyAV/NVDEC)      (ONNX + TensorRT)      (CuPy, GPU-only)      (aiortc)        (Telemetry UI)
```

---

## Key Results

| Metric | Result |
|---|---|
| TensorRT vs. native PyTorch speedup | **3.05x** (YOLOv8s, FP16, raw forward pass vs. raw engine execution) |
| Zero-copy pipeline latency (avg) | 3.99 ms |
| Zero-copy pipeline latency (P95) | 16.02 ms |
| Effective inference throughput | ~250 FPS (single stream, RTX 3050 Laptop GPU, 4GB VRAM) |

Benchmarked on: NVIDIA GeForce RTX 3050 Laptop GPU · CUDA 12.7 · TensorRT 11.1.0 · PyTorch 2.5.1+cu121

---

## Project Structure

```
Visionedge/
├── backend/
│   ├── decoder/          # Hardware/CPU video decoding (PyAV, NVDEC, fallback paths)
│   ├── inference/         # Model export, TensorRT engine build, benchmarking
│   │   ├── model_compile.py       # PyTorch → ONNX export (Ultralytics pipeline)
│   │   ├── engine.py              # ONNX → TensorRT engine compilation
│   │   ├── inference_loop.py      # TensorRT inference loop + latency measurement
│   │   ├── benchmark_pytorch.py   # Raw PyTorch forward-pass FPS benchmark
│   │   ├── benchmark_tensorrt.py  # Raw TensorRT engine FPS benchmark
│   │   └── benchmark_compare.py   # Side-by-side speedup comparison
│   ├── pipeline/
│   │   └── zero_copy.py           # CuPy-based zero-copy GPU pipeline
│   ├── streaming/
│   │   ├── webrtc_server.py       # aiortc WebRTC signaling server
│   │   ├── file_video_track.py    # Streams a local video file over WebRTC
│   │   └── metrics.py             # Live telemetry endpoint (FPS, GPU memory %)
│   └── utils/                     # Shared config, logging, constants
│
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── VideoStream.jsx         # WebRTC video viewer
│           └── TelemetryDashboard.jsx  # Live pipeline metrics UI
│
├── models/                # Exported .pt / .onnx / .engine model artifacts
├── tests/
└── docs/
```

---

## Setup

### Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA 12.x drivers installed
- Node.js (for the frontend)

### Backend

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics onnx tensorrt-cu12 --extra-index-url https://pypi.nvidia.com
pip install cupy-cuda12x aiortc aiohttp opencv-python av
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Run the pipeline

```bash
# 1. Export a YOLO model to ONNX
python backend/inference/model_compile.py

# 2. Compile the ONNX model into a TensorRT engine
python backend/inference/engine.py

# 3. Benchmark PyTorch vs. TensorRT
python backend/inference/benchmark_compare.py

# 4. Run the zero-copy GPU pipeline
python backend/pipeline/zero_copy.py

# 5. Start the WebRTC streaming server
python -m backend.streaming.webrtc_server
```

Then open the frontend (`http://localhost:5173` by default) to view the live stream and telemetry dashboard.

**Note:** drop a sample `.mp4` file at `backend/streaming/sample.mp4` (or set the `VIDEO_PATH` environment variable) before starting the streaming server — sample videos aren't committed to the repo.

---

## Team

| Member | Branch | Focus |
|---|---|---|
| Mohd Abdul Razzaq (Lead) | `razzaq` | Model compilation, TensorRT engine, zero-copy pipeline, benchmarking |
| Ritam Halder | `ritam` | Hardware video decoding (NVDEC) |
| Srinivas | `srinivas` | Inference loop, latency measurement, telemetry dashboard |
| Usman | `usman` | WebRTC streaming (React + aiortc) |
| Asha | `asha` | Decoder support (CPU/QSV fallback path) |
| Bhargavi | `bhargavi` | Shared config/utils module |

---

## Git Workflow

- Each member works on their own branch and commits directly to it.
- `main` is protected — all changes land via Pull Request, requiring review before merge.
- See `docs/` for the team's Work Distribution Document and progress notes.

---

## Status

Actively in development — Week 1 through Week 3 deliverables complete for the core inference/GPU pipeline. See `docs/week_progress.md` for detailed milestone notes.
