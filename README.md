# High-Performance Face Recognition — README

> Optimized real-time face recognition combining **RetinaFace** (detector) + **ArcFace** (re-identification/embedding) + a customized **ByteTrack** tracker — all served with **TensorRT** engines and Python `multiprocessing` for maximum throughput.
> Capable of ~**80 FPS** on a properly provisioned NVIDIA GPU (see *Performance & Tuning*).

![Alt Text](readme.gif)

---

## Table of contents

* [Key features](#key-features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Running Qdrant (vector DB)](#running-qdrant-vector-db)
* [Build TensorRT engine files](#build-tensorrt-engine-files)
* [Configuration / `.env` example](#configuration--env-example)
* [Running the project](#running-the-project)
* [Performance & tuning tips](#performance--tuning-tips)
* [Minimum Requirment & Run tests](#minimum--requirments)
* [Architecture overview](#architecture-overview)
* [Contributing](#contributing)

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

4. **Insert vector**

after running Qdrant, you can use `qdrantInsertPerson.ipynb` to insert persons into qdrant database.

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

1. download retinaface`RetinaFace-R50.pth` and arcface `arcface-r100-glint360k.pth` files from any source you can.
- https://huggingface.co/camenduru/video-retalking/blob/main/RetinaFace-R50.pth
- https://huggingface.co/BooBooWu/Vec2Face/blob/c701064b38ef354f9d56303d6efb6afec8023dd4/weights/arcface-r100-glint360k.pth

1. make onnx models from .pt model files from `testbench/create_engine.ipynb`.

2. create tensorrt engines with commands below at `checkpoints/`:
```bash
cd checkpoints

# RetinaFace
!trtexec  --onnx=RetinaFace-R50.onnx --saveEngine=RetinaFace-R50_fp16.engine --fp16

# ArcFace
!trtexec  --onnx=arcface-r100-glint360k.onnx --saveEngine=arcface-r100-glint360k_fp16.engine --fp16
```

Notes:

* Use `--fp16` for faster inference where supported; use `--int8` only if you provide a calibration dataset.
* The exact conversion flow depends on your ONNX export settings — make sure the ONNX opset and layer compatibility are correct.

---

## Configuration / `.env` example

Create a `.env` in the project root (or set equivalent environment variables):

```env
# .env example
VIDEO_SOURCE=http://192.168.1.102:4747/video   # or a file path like "test_madrid.mov" or "0" for default webcam or
VIDEO_SOURCE=rtsp://{user}:{password}@{host}:{port}/Stream/Live/101
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

## Running the project

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

(If you are on Linux/Mac, use the project’s provided run script or run the Python entrypoint directly, e.g. `python -m app.main` — check your repository's `pyproject.toml` or `scripts/` directory.)

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

  * Measure per-stage latency (capture → detection → candidates → recognition → gui) to identify bottlenecks.

---

## Minimum Requirements & Run Tests

### Minimum hardware requirements
To run the system reliably in real time, you need at least:

- **GPU:** NVIDIA GPU with **3 GB** of CUDA VRAM (minimum for RetinaFace + ArcFace TensorRT engines)  
- **CPU:** Modern quad-core CPU (more cores improve multiprocessing throughput)  
- **RAM:** 8 GB system memory (more recommended for multi-stream processing)  
- **Disk:** SSD recommended for fast video loading and model initialization  
- **Software:**  
  - CUDA-compatible NVIDIA driver  
  - TensorRT installed  
  - Python 3.9+  
  - Docker (for Qdrant)

---

### Performance benchmarks

Measured on **HD video (1280×720)** using FP16 TensorRT engines:

| CPU                  | GPU                       | GPU VRAM | FPS (HD Video)                   | Notes |
|----------------------|----------------------------|----------|----------------------------------|-------|
| **Core i5-4790K**    | **GTX 1660 Ti (6 GB)**     | 6 GB     | **~40 FPS**                      | Stable real-time performance |
| **Core i7-14700**    | **RTX 4080 SUPER (16 GB)** | 16 GB    | **~60 FPS (capped)**             | System can exceed **100+ FPS**, capped at **60 FPS** for stability |

> High-end GPUs can push well beyond 100 FPS. For production workloads, capping FPS (e.g., 60–80) helps maintain consistent latency and overall system stability, you can inform me from your own tests.


---

## Architecture overview


```
[Capture Process] --> frames queue -->
[Detection Process (TensorRT RetinaFace)] --> detections queue -->
[Detection Post Process (NMS + other things)] --> candidates queue -->
[Recognition Process (TensorRT ArcFace + ByteTrack)] --> recognition queue -->
                                      \
                                       --> [Qdrant client] (store / query embeddings)
[GUI Process] --> gui queue                                   
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
3. Submit a pull request with a clear description of changes

Add unit tests for new logic paths where possible.
