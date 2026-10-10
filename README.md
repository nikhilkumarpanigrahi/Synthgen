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

- 🛣️ **Civil Infrastructure:** Road damage inspection (`pothole`, `surface_crack`).
- 🔬 **Dermatology & Medical Imaging:** Skin lesion classification (`melanoma`, `nevus`, `seborrheic_keratosis`).
- 🏭 **Industrial Manufacturing:** NEU steel defect inspection (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled_in_scale`, `scratches`).
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

On Windows PowerShell, use the project interpreter:

```powershell
.\.venv\Scripts\Activate.ps1
python server.py
```

Open your browser at **[http://localhost:8080](http://localhost:8080)**. Set `$env:PORT = "8081"` first if port 8080 is unavailable. Interactive REST API documentation is available at `/docs` on the selected port.

### 3. Deploy to Google Cloud Run

The included `Dockerfile` runs FastAPI with Uvicorn on the `PORT` supplied by Cloud Run. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), create or select a billed GCP project, and authenticate:

```powershell
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

Set deployment variables in PowerShell:

```powershell
$env:PROJECT_ID = "YOUR_PROJECT_ID"
$env:REGION = "us-central1"
$env:REPOSITORY = "synthgen"
$env:SERVICE = "synthgen-web"
$env:IMAGE = "$env:REGION-docker.pkg.dev/$env:PROJECT_ID/$env:REPOSITORY/$env:SERVICE:v1"
```

Create the Docker repository once, then build and push the image from the repository root:

```powershell
gcloud artifacts repositories create $env:REPOSITORY --repository-format=docker --location=$env:REGION --description="Synthgen container images"
gcloud builds submit --tag $env:IMAGE .
```

Deploy the initial service. These settings keep expensive in-process PyTorch jobs from multiplying across instances:

```powershell
gcloud run deploy $env:SERVICE `
   --image $env:IMAGE `
   --region $env:REGION `
   --platform managed `
   --cpu 4 `
   --memory 8Gi `
   --timeout 3600 `
   --concurrency 1 `
   --max 1 `
   --set-env-vars PYTHONUNBUFFERED=1 `
   --allow-unauthenticated
```

Use `--no-allow-unauthenticated` for a private service. Retrieve and verify the HTTPS URL:

```powershell
$env:SERVICE_URL = gcloud run services describe $env:SERVICE --region $env:REGION --format="value(status.url)"
Invoke-WebRequest "$env:SERVICE_URL/" -UseBasicParsing
Invoke-WebRequest "$env:SERVICE_URL/docs" -UseBasicParsing
Invoke-WebRequest "$env:SERVICE_URL/api/system/status" -UseBasicParsing
```

This is suitable for an initial demo. Cloud Run's local filesystem is ephemeral, while this application currently stores uploaded datasets, generated images, models, and active-dataset state under `data/`. Move those assets to Cloud Storage before relying on the deployment for durable research data. Training endpoints also run synchronously in the web process; use Cloud Run Jobs or a queue-backed worker for production-scale training.

### 4. Load Real Datasets

Synthgen keeps all real images in its existing `data/<domain>/real/<class>/` folders. The local import command replaces only the selected domain's real images, activates that domain, and leaves the shared generation, QA, and classifier pipeline unchanged.

From PowerShell, use the extracted dataset root as `--source`:

```powershell
python server.py import-local --domain road_defects --source "C:\datasets\road"
python server.py import-local --domain industrial_defects --source "C:\datasets\NEU"
```

Road class folders are mapped to `pothole` and `surface_crack`. NEU class folders are preserved as `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled_in_scale`, and `scratches`. The importer searches recursively and accepts JPG, JPEG, and PNG files.

For skin lesions, Synthgen uses the official ISIC API rather than downloading the ISIC 2017 archive:

```powershell
python server.py download-isic --per-class 2
python server.py download-isic --per-class 20
```

This calls `/api/v2/images/search/` with diagnosis queries for melanoma, nevus, and seborrheic keratosis, then downloads only each result's `files.full.url`. Existing ISIC IDs are skipped. Public access is sufficient; if the API requires authentication in your environment, configure the token without putting it in source control:

```powershell
$env:ISIC_API_TOKEN = "your-token"
python server.py download-isic --per-class 20
```

The resulting folders are:

```text
data/
   road_defects/real/pothole/
   road_defects/real/surface_crack/  # required for two-class road experiments
   medical_imaging/real/melanoma/
   medical_imaging/real/nevus/
   medical_imaging/real/seborrheic_keratosis/
   industrial_defects/real/crazing/
   industrial_defects/real/inclusion/
   industrial_defects/real/patches/
   industrial_defects/real/pitted_surface/
   industrial_defects/real/rolled_in_scale/
   industrial_defects/real/scratches/
```

Check each imported domain before training:

```powershell
python -c "from backend.dataset_manager import inspect_dataset_inventory as i; print(i('road_defects')); print(i('medical_imaging')); print(i('industrial_defects'))"
```

Then start the application, select each domain in the dashboard, inspect its real samples, run a small generation job for a discovered class, and run the benchmark. The benchmark uses the existing 75/25 real train/validation split and the same preprocessing for all domains.

### 4. Verified Local Dataset State

The current workspace has been verified with these real-image counts:

| Domain | Classes | Real images |
| --- | --- | ---: |
| Road | `pothole` | 210 |
| Skin | `melanoma`, `nevus`, `seborrheic_keratosis` | 20 each |
| Steel | Six NEU defect classes | 300 each |

The current Road archive did not contain a crack class, so `surface_crack` is not populated yet. Add a directory containing crack images and re-run the Road import before benchmarking road damage classification.
---

## 🎨 UI Aesthetic

Designed specifically for machine learning practitioners and researchers:
- **Obsidian Matte Black** palette (`#000000`, surfaces `#0A0A0A`, borders `#222222`).
- High-contrast monochrome typography with zero neon colors and zero decorative animations.
- Real-time streaming Thought Trace console.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
