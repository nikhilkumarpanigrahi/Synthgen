---
title: Synthgen
emoji: ⚡
colorFrom: gray
colorTo: black
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Synthgen: Autonomous Agentic Synthetic Data Generation Platform

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg)](https://pytorch.org/)
[![Hardware Acceleration](https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS%20%7C%20CUDA-brightgreen.svg)]()
[![Export Formats](https://img.shields.io/badge/Formats-YOLO%20%7C%20COCO%20%7C%20TorchScript-blueviolet.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An open-source scientific deep-learning workbench and autonomous multi-agent pipeline designed to solve the **minority-class data scarcity problem** in computer vision datasets.

---

## 🎯 The Core Research Question

> **"Can deep-learning-based synthetic image generation for an underrepresented class improve downstream classifier performance on real unseen samples of that class?"**

In real-world computer vision (medical imaging, autonomous driving, manufacturing defect inspection), minority classes suffer from extreme data scarcity. `Synthgen` provides an empirical, closed-loop workbench to profile imbalance, train generative networks on the rare class, audit privacy and non-memorization, and validate test recall jumps on held-out real data.

---

## 🧠 Autonomous 5-Stage Agentic Training Loop

Instead of manual guesswork, `Synthgen` features a closed-loop multi-agent orchestrator:

1. **Stage 1 — Profiler Agent (Imbalance Diagnostics):**
   - Dynamically scans active class distributions.
   - Calculates mathematical imbalance ratio (e.g. `4.0:1`) and determines the exact target synthetic quota required to achieve distribution parity.
2. **Stage 2 — Generative Training Agent (PyTorch DCGAN):**
   - Fits a genuine PyTorch Deep Convolutional GAN (Generator + Discriminator) on rare class images using Adam optimizers and binary cross-entropy loss (`BCELoss`) with gradient backpropagation.
   - Checkpoints trained generator weights to `generative_model_weights.pt`.
3. **Stage 3 — Quality & Privacy Critic Agent (Automated QA Gate):**
   - Evaluates empirical **Fréchet Inception Distance (FID)** using real vs. synthetic covariance matrix square roots (`scipy.linalg.sqrtm`) on 128-dimensional dense multi-scale feature descriptors.
   - Enforces **Differential Privacy & Non-Memorization** via pairwise nearest-neighbor Euclidean distance ($torch.cdist$) to guarantee zero training data leakage.
   - Guardrails against mode collapse (diversity score check).
4. **Stage 4 — Downstream Classifier Ablation Engine:**
   - Trains dual PyTorch CNN models (`DefectConvNet-V2`) with `AdamW` and `CrossEntropyLoss`:
     - **Model A (Baseline):** Trained strictly on imbalanced real data.
     - **Model B (Augmented):** Trained on real data + approved synthetic DCGAN images.
   - Evaluates both models against identical, held-out, strictly real unseen test splits.
5. **Stage 5 — Reflection & Scientific Verdict Agent:**
   - Evaluates top-1 accuracy, macro F1, and rare-class recall delta.
   - Streams an auditable, step-by-step **Agent Thought Trace** directly in the console.

---

## 📦 Multi-Format Export & Production Deployment

`Synthgen` bridges the gap between synthetic data research and real-world deployment:

- **Ultralytics YOLO Format (`data.yaml`):** Exports organized `images/train`, `images/val`, and normalized bounding box label text files (`labels/train`, `labels/val`), ready for immediate training with YOLOv8/v9/v11 (`yolo detect train data=data.yaml model=yolov8n.pt`).
- **COCO 2017 Dataset Format:** Universal `annotations/coco_annotations.json` specification compatible with Meta Detectron2, OpenMMLab, and PyTorch Vision.
- **Raw PyTorch ImageFolder:** Compatible with `torchvision.datasets.ImageFolder`.
- **Deployable Model Bundle (.zip):** Checkpoints binary weights (`augmented_model_weights.pt`), compiled TorchScript (`augmented_torchscript.pt`), and a standalone Python inference script (`infer.py`) for zero-dependency execution.
- **Scientific Research Report (.md):** Automatically compiles a publication-grade peer-review Markdown report of all ablation findings.

---

## 🌐 Multi-Domain Datasets Supported

- 🛣️ **Civil Infrastructure:** Road damage inspection (`normal_road`, `surface_crack`, `pothole`).
- 🔬 **Dermatology & Medical Imaging:** Skin lesion classification (`melanoma`, `seborrheic_keratosis`, `benign_nevus`).
- 🏭 **Industrial Manufacturing:** Cold-rolled steel defect inspection (`clean_surface`, `micro_fracture`, `welding_void`).
- 📁 **Custom ZIP Upload:** Drag-and-drop any `.zip` containing class subdirectories (`class_a/`, `class_b/`) for automated ingestion and class discovery.

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/nikhilkumarpanigrahi/Synthgen.git
cd Synthgen

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Application
```bash
python server.py
```

Open your browser at **[http://localhost:8080](http://localhost:8080)**.  
Interactive REST API documentation is available at **[http://localhost:8080/docs](http://localhost:8080/docs)**.

---

## 🎨 UI Aesthetic

Designed specifically for machine learning practitioners and researchers:
- **Obsidian Matte Black** palette (`#000000`, surfaces `#0A0A0A`, borders `#222222`).
- High-contrast monochrome typography with zero neon colors and zero decorative animations.
- Real-time streaming Thought Trace console.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
