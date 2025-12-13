M3R — Multimodal 3D Reconstruction & Rendering System

M3R is a reproducible, end-to-end prototype for multimodal 3D reconstruction and visualization, combining 3D deep learning, spatial representations, and interactive rendering in a single system.

The project is designed to demonstrate practical engineering competence in 3D perception and visual computing, with an emphasis on medical imaging–style volumetric data and spatial AI workflows.

This repository is intentionally scoped as a working system prototype, not a benchmark-focused research paper.

What This Project Demonstrates

End-to-end 3D reconstruction from sparse volumetric input

Gaussian splatting–style rendering for fast visualization

Implicit (NeRF-inspired) density representations (demo-level)

Reproducible experiments with deterministic data generation

API-based inference and rendering using FastAPI

Clean separation between data, models, rendering, and deployment

System Overview

Pipeline:

Synthetic 3D volumetric data generation (NIfTI format)

Sparse sampling of the volume (simulating limited acquisition)

3D convolutional reconstruction model (autoencoder / UNet-style)

Visualization via:

Gaussian splatting views

Implicit density (NeRF-style) projections

Deployment via FastAPI endpoints for interactive use

Data → Sparse Sampling → 3D Model → Rendering → API

Components
1. 3D Reconstruction Model

Lightweight 3D autoencoder (UNet-style baseline)

Trained on volumetric data with sparse slice input

Produces reconstructed 3D volumes

Evaluated using PSNR and SSIM

This model serves as a baseline reconstruction component and is intentionally kept simple and CPU-friendly.

2. Gaussian Splatting Renderer

Converts volumetric intensity distributions into point-based splats

Generates multiple camera-angle views

Deterministic and fast to render

Useful for intuitive visualization of 3D structure

3. Implicit Density Representation (NeRF-style)

Lightweight NeRF-inspired implicit density module

Used to demonstrate implicit spatial representations

The current visualization is contrast-enhanced for interpretability

Full NeRF training is outside the scope of this prototype

This component is included to show architectural extensibility, not to compete with trained NeRF pipelines.

4. FastAPI Deployment

Interactive API endpoints expose the system:

Endpoint	Description
/health	Service health check
/reconstruct	Run model inference and return reconstructed slice
/render/gauss	Gaussian splatting visualization
/render/nerf_density	Implicit density visualization

Swagger UI available at:

http://127.0.0.1:8000/docs

Running the Project
1. Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

2. Generate Data
python scripts/create_synthetic_data.py

3. Train Reconstruction Model
python scripts/train_unet.py

4. Generate Renderings
python scripts/run_splat_demo.py

5. Start API
python -m uvicorn src.app.api:app --host 127.0.0.1 --port 8000

Why This Project Matters
Medical Imaging

Sparse-to-dense reconstruction mirrors challenges in MRI / CT acceleration

Volumetric reasoning and spatial coherence are core requirements

Emphasizes interpretability and reproducibility over black-box metrics

Spatial AI & Robotics

Demonstrates 3D scene representation and rendering

Modular design allows integration with sensors, simulators, or SLAM pipelines

Highlights the trade-offs between explicit (voxels/splats) and implicit (NeRF-style) representations

Engineering Principles Emphasized

Deterministic data generation

Clear module boundaries

CPU-friendly defaults

Explicit testing of data, model, rendering, and API layers

Honest separation between production-ready and demo components

Limitations & Future Work

Replace baseline autoencoder with a full UNet or MONAI-based architecture

Train implicit models on real datasets (e.g., fastMRI slices)

Multi-view NeRF training and differentiable rendering

GPU acceleration and ONNX export

Front-end viewer for interactive 3D exploration

Repository Structure
M3R/
├── data/                  # Generated volumes and outputs
├── models/                # Trained model weights
├── scripts/               # Training, rendering, utilities
├── src/
│   ├── app/               # FastAPI app
│   ├── models/            # 3D reconstruction models
│   └── splatting/         # Rendering logic
└── README.md

Author Notes

This project was built to explore how different 3D representations interact in a real system, rather than optimizing a single algorithm in isolation. The emphasis is on end-to-end thinking, engineering clarity, and spatial reasoning.