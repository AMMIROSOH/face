# High-Performance Face Recognition — README

> Optimized real-time face recognition combining **RetinaFace** (detector) + **ArcFace** (re-identification/embedding) + a customized **ByteTrack** tracker — all served with **TensorRT** engines and Python `multiprocessing` for maximum throughput.
> Capable of ~**80 FPS** on a properly provisioned NVIDIA GPU (see *Performance & Tuning*).

![GIF placeholder](./assets/placeholder.gif)
*(Replace `./assets/placeholder.gif` with your demo GIF when ready.)*

---

## Table of contents

* [Key features](#key-features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Running Qdrant (vector DB)](#running-qdrant-vector-db)
* [Build TensorRT engine files](#build-tensorrt-engine-files)
* [Running the project](#running-the-project)
* [Configuration / `.env` example](#configuration--env-example)
* [Usage examples](#usage-examples)
* [Performance & tuning tips](#performance--tuning-tips)
* [Architecture overview](#architecture-overview)
* [Troubleshooting](#troubleshooting)
* [Contributing](#contributing)
* [License](#license)

---

## Key features

* RetinaFace detector converted to TensorRT for low-latency detection.
* ArcFace embedding model converted to TensorRT for ultra-fast feature extraction.
* Customized ByteTrack for robust multi-person tracking and ID persistence.
* Python `multiprocessing` isolates heavy tasks (capture → detection → filtering(nms) → recognition → draw GUI) into separate processes to leverage multi-core CPUs while keeping GPU usage efficient.
* Supports video files, RTSP/HTTP streams, and webcam input.
* Integrates with Qdrant for vector indexing and fast nearest-neighbor search.

---

## Prerequisites

* Linux or Windows with a compatible NVIDIA GPU and CUDA installed.
* Matching CUDA runtime / driver for the TensorRT version you will use.
* Python 3.10+ (recommend using Poetry-managed virtualenv).
* `poetry` (project uses Poetry for dependency management).
* TensorRT (to build and run `.engine` files).
* Docker (for running Qdrant in a container).

---

## Installation

1. **Install correct PyTorch + CUDA wheel**
   Adjust the `cu126` part to match your local CUDA version if necessary.

```bash
# Example (provided in your project)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# Or check https://pytorch.org/get-started/locally/ for the correct command for your CUDA
```

2. **Install PyCUDA** (used by TensorRT runtime wrappers or some helper scripts)

```bash
pip install pycuda
```

3. **Install project dependencies with Poetry**

```bash
# Ensure poetry is installed on your machine
poetry install
# Activate the virtual environment
poetry shell
```

> If you prefer `pip`/`venv` you can adapt, but the repository uses Poetry by default.

---

## Running Qdrant (vector DB)

Run a Qdrant container to store and search face embeddings. The container exposes the internal Qdrant port (`6333`) — map it to the host port you prefer (your `.env` in examples uses host port `6334`).

```bash
# map host:container ports so host 6334 forwards to container 6333
docker run -d --name qdrant \
  -p 6334:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

If you prefer the default host port `6333`, change the `-p` mapping and `.env` accordingly.

---

## Build TensorRT engine files

The project expects TensorRT `.engine` files for both RetinaFace and ArcFace. You can create them from ONNX (or another intermediate) using `trtexec` or the TensorRT Python API.

Example using `trtexec` (adjust paths / options for your models):

```bash
# RetinaFace (example)
trtexec --onnx=retinaface.onnx --saveEngine=retinaface.engine --fp16 --workspace=2048 --explicitBatch

# ArcFace (example)
trtexec --onnx=arcface.onnx --saveEngine=arcface.engine --fp16 --workspace=2048 --explicitBatch
```

Notes:

* Use `--fp16` for faster inference where supported; use `--int8` only if you provide a calibration dataset.
* `--workspace` (MB) may need to increase for large models or optimizations.
* The exact conversion flow depends on your ONNX export settings — make sure the ONNX opset and layer compatibility are correct.

Place the generated `.engine` files into your project `models/` directory or point the config to their locations.

---

## Running the project

There is a convenience PowerShell script `dev.ps1` to start the project in development mode.

From PowerShell:

```powershell
# run the dev helper
.\dev.ps1
```

(If you are on Linux/Mac, use the project’s provided run script or run the Python entrypoint directly, e.g. `python -m app.main` — check your repository's `pyproject.toml` or `scripts/` directory.)

The project accepts both a video file or a network/webcam stream as input.

---

## Configuration / `.env` example

Create a `.env` in the project root (or set equivalent environment variables):

```env
# .env example
VIDEO_SOURCE=http://192.168.1.102:4747/video   # or a file path like "test_madrid.mov" or "0" for default webcam
QDRANT_HOST=localhost
QDRANT_PORT=6334
# Optional tuning vars you may find in the project:
# DETECTOR_ENGINE=models/retinaface.engine
# EMBEDDING_ENGINE=models/arcface.engine
# FPS_TARGET=80
# USE_FP16=true
```

The app reads `VIDEO_SOURCE` and Qdrant connection info from the environment at startup.

---

## Usage examples

* Feed from an IP camera or smartphone stream:

```bash
# With .env set to VIDEO_SOURCE=http://192.168.1.102:4747/video
.\dev.ps1
```

* Feed from a local file:

```bash
# Example run passing CLI override (if supported)
python app/main.py --source "test_madrid.mov"
```

* Use webcam (index `0`):

```bash
export VIDEO_SOURCE="0"
.\dev.ps1
```

---

## Performance & tuning tips

* **Batch size**: For real-time single-frame processing keep batch size = 1. For multi-frame batching you may increase batch but latency changes.
* **Precision**: Use **FP16** engines where your GPU supports it (significant speed up). Use INT8 only with a proper calibration dataset.
* **Workspace**: Increase TensorRT workspace when building engines (`--workspace` in `trtexec`) to allow better optimizations.
* **Multiprocessing & CUDA**:

  * Use `multiprocessing.set_start_method('spawn')` on platforms (Windows, macOS) where required to avoid CUDA context issues.
  * Create CUDA contexts only in worker processes that need GPU access. Avoid forking a process after the CUDA context is created.
* **I/O**:

  * Use a separate process for frame capture + queue buffering to keep GPU workers fed.
  * Pin memory and use fast transforms (avoid expensive Python image conversions in the hot path).
* **Tracker**:

  * ByteTrack parameters (e.g., detection confidence thresholds, embedding distance thresholds) drastically affect both performance and track stability. Tune them based on your scenario.
* **Profiling**:

  * Measure per-stage latency (capture → preprocess → detect → embed → track → DB write) to identify bottlenecks.

---

## Architecture overview

Simple pipeline (process boundaries may vary per your implementation):

```
[Capture Process] --> frames queue -->
[Detection Process (TensorRT RetinaFace)] --> detections queue -->
[Embedding Process (TensorRT ArcFace)] --> embeddings queue -->
[Tracking Process (ByteTrack + matching)] --> results
                                      \
                                       --> [Qdrant client] (store / query embeddings)
```

Each stage runs in its own Python process to use multiple CPU cores while keeping the GPU busy with inference.

---

## Troubleshooting

* **Out of GPU memory / OOM**

  * Reduce workspace when building engines. Use FP16 or smaller batch size. Make sure no other processes are holding large GPU memory (e.g., leftover Python interpreters).
* **TensorRT engine load errors**

  * Ensure the engine was built for the same TensorRT version and compute capability as the runtime environment.
* **Qdrant connectivity issues**

  * Verify `docker ps` shows Qdrant running. Double-check the `-p` mapping and `.env` host/port values.
* **Multiprocessing + CUDA errors**

  * Use the `spawn` start method. Avoid creating CUDA contexts before forking processes.
* **Lower-than-expected FPS**

  * Profile each stage. Possible causes: slow capture, excessive preprocessing, single-threaded I/O, or CPU-bound tracker logic.

---

## Contributing

Contributions are welcome. Please open issues for bugs or feature requests. For code contributions:

1. Fork the repository
2. Create a topic branch
3. Run `poetry install` and ensure tests pass (if present)
4. Submit a pull request with a clear description of changes

Add unit tests for new logic paths where possible.

---

## TODO / Roadmap

* Replace GIF placeholder with a short demo GIF / video.
* Add INT8 calibration pipeline for further acceleration.
* Add a lightweight web UI for live preview and manual labeling.
* Add automated benchmark script with common hardware profiles.

---

## License

Specify your license here (e.g., MIT). Example:

```
MIT License
Copyright (c) YEAR Your Name
```

---

If you want, I can:

* generate a polished GIF placeholder and add the correct markdown snippet, or
* create a `scripts/build_engine.sh` and `scripts/run_qdrant.sh` helper files with the exact commands used above, or
* produce a short CONTRIBUTING.md or CHANGELOG.md to go with this README.

Which of those would you like me to add next?
