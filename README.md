# Synthetix ML — Synthetic Data Engine & QA Platform

An enterprise-grade platform for synthesizing, validating, and benchmarking computer vision datasets for training defect detection models.

---

## Capabilities & Architecture

1. **Dataset Explorer & Ingestion**
   - High-throughput indexing of real road damage images across defect classes (`pothole`, `surface_crack`, `normal_road`).
   - Imbalance ratio telemetry (`1 : 2.50` severe minority bias on rare defect classes).
   - Direct image drag-and-drop ingestion with automatic label attribution.
   - Deep image inspector with resolution, byte size, and class descriptors.

2. **Generative Model Dispatcher**
   - Multi-architecture tensor engine:
     - **Latent Diffusion (DDPM)** with cosine noise schedule and classifier-free guidance (CFG).
     - **StyleGAN-v2** with adversarial discriminator scoring.
     - **Beta-VAE** with disentangled latent representations.
     - **Auto-Augment** with affine and photometric transformations.
   - Asynchronous job execution pipeline with queue management, progress tracking, and execution logs.

3. **Statistical Quality Assurance & Privacy Audit**
   - **Fréchet Inception Distance (FID)**: Automated distribution divergence calculation.
   - **Inception Score (IS)**: Empirical conditional class distribution entropy.
   - **Perceptual Feature Diversity**: Pairwise feature variance to guarantee absence of mode collapse.
   - **Differential Privacy & Memorization Check**: Nearest-neighbor Euclidean distance validation ensuring generated samples do not duplicate real training data.

4. **Downstream Classifier Benchmarks**
   - Trains and compares ResNet-18 / ConvNet architectures:
     - **Model A (Baseline)**: Real data only (imbalanced, scarce).
     - **Model B (Augmented)**: Real + Synthetic balanced dataset.
   - Produces Scikit-Learn style classification reports (Precision, Recall, F1-Score, Support) and Confusion Matrices.
   - Interactive training convergence loss curves and validation accuracy tracking across epochs.

5. **Dataset Packaging & Export**
   - One-click export of datasets with standardized directory manifests and metadata JSON.

---

## Quick Start

```bash
cd /Users/nikhilkumarpanigrahi/.gemini/antigravity-ide/scratch/synthetic-data-generator
./start.sh
```

- **Web Dashboard**: [http://localhost:8080](http://localhost:8080)
- **REST API Specs**: [http://localhost:8080/docs](http://localhost:8080/docs)
