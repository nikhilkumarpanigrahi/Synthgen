import os
import io
import json
import zipfile
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .dataset_manager import (
    initialize_domain_datasets,
    get_available_datasets,
    inspect_dataset_inventory,
    get_dataset_samples,
    set_active_dataset_id,
    get_active_dataset_id,
    get_dataset_base_paths,
    ingest_custom_zip,
    DATA_DIR
)
from .generator import GenerationPipeline, JOBS_REGISTRY
from .evaluator import QualityAssuranceEngine
from .cnn_trainer import DownstreamClassifierBenchmark

app = FastAPI(
    title="Synthetic Data Generator Platform",
    description="Scientific Deep-Learning Synthetic Data Generation & Minority-Class Augmentation Platform",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = GenerationPipeline()
qa_engine = QualityAssuranceEngine()
benchmark_engine = DownstreamClassifierBenchmark()

class DatasetSelectRequest(BaseModel):
    dataset_id: str

class JobCreateRequest(BaseModel):
    architecture: str = "diffusion"  # diffusion, gan, vae, augmentation
    target_class: str
    count: int = 8
    cfg_scale: float = 7.5
    seed: int = 42
    resolution: int = 256

class BenchmarkRunRequest(BaseModel):
    backbone: str = "DefectConvNet-V2"
    epochs: int = 12
    learning_rate: float = 0.002

@app.on_event("startup")
def startup_event():
    # Bootstrap multi-domain datasets if not present
    initialize_domain_datasets()

@app.get("/api/system/status")
def system_status():
    import torch
    dev = "Apple Silicon MPS (Metal)" if torch.backends.mps.is_available() else ("CUDA" if torch.cuda.is_available() else "CPU")
    return {
        "status": "HEALTHY",
        "version": "2.1.0",
        "engine": f"PyTorch {torch.__version__} ({dev})",
        "storage": DATA_DIR,
        "active_dataset": get_active_dataset_id()
    }

@app.get("/api/datasets/available")
def list_available_datasets():
    return get_available_datasets()

@app.post("/api/datasets/select")
def select_dataset(req: DatasetSelectRequest):
    set_active_dataset_id(req.dataset_id)
    initialize_domain_datasets()
    return inspect_dataset_inventory(req.dataset_id)

@app.post("/api/datasets/upload-zip")
async def upload_custom_dataset_zip(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip archives containing class subdirectories are supported.")
    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Archive exceeds maximum size limit of 100MB.")
    
    result = ingest_custom_zip(content, file.filename)
    return {"status": "SUCCESS", "dataset": result}

@app.get("/api/dataset/inventory")
def dataset_inventory():
    return inspect_dataset_inventory()

@app.get("/api/dataset/samples")
def dataset_samples(
    category: str = Query("real", enum=["real", "synthetic"]),
    class_name: Optional[str] = None,
    limit: int = Query(40, le=200),
    offset: int = Query(0, ge=0)
):
    return get_dataset_samples(category=category, class_name=class_name, limit=limit, offset=offset)

@app.post("/api/jobs/create")
def create_job(req: JobCreateRequest):
    inv = inspect_dataset_inventory()
    if req.target_class not in inv["classes"]:
        raise HTTPException(status_code=400, detail=f"Invalid class '{req.target_class}'. Available classes in active dataset: {inv['classes']}")
    
    job_id = pipeline.create_job(
        architecture=req.architecture,
        target_class=req.target_class,
        count=req.count,
        cfg_scale=req.cfg_scale,
        seed=req.seed,
        resolution=req.resolution
    )
    return {"job_id": job_id, "status": "QUEUED"}

@app.post("/api/jobs/{job_id}/run")
def run_job(job_id: str):
    if job_id not in JOBS_REGISTRY:
        raise HTTPException(status_code=404, detail="Job not found")
    result = pipeline.execute_job(job_id)
    return result

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    job = JOBS_REGISTRY.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/api/quality/audit")
def quality_audit():
    return qa_engine.run_qa_audit()

@app.post("/api/benchmark/train")
def run_benchmark(req: BenchmarkRunRequest):
    return benchmark_engine.run_experiment(
        backbone=req.backbone,
        epochs=req.epochs,
        learning_rate=req.learning_rate
    )

class AgenticRunRequest(BaseModel):
    gan_epochs: int = 20
    classifier_epochs: int = 8

@app.post("/api/agentic/run")
def trigger_agentic_training(req: AgenticRunRequest = AgenticRunRequest()):
    from .agentic_trainer import agentic_orchestrator
    return agentic_orchestrator.run_autonomous_pipeline(
        gan_epochs=req.gan_epochs,
        classifier_epochs=req.classifier_epochs
    )

@app.get("/api/agentic/status")
def get_agentic_status():
    from .agentic_trainer import agentic_orchestrator
    return {
        "is_running": agentic_orchestrator.is_running,
        "current_stage": agentic_orchestrator.current_stage,
        "thought_trace": agentic_orchestrator.thought_trace,
        "latest_result": agentic_orchestrator.latest_result
    }

@app.get("/api/benchmark/report")
def get_benchmark_report():
    from .cnn_trainer import LAST_BENCHMARK_RESULT, generate_markdown_report
    res = LAST_BENCHMARK_RESULT
    if not res:
        res = benchmark_engine.run_experiment(epochs=8)
    report_md = generate_markdown_report(res)
    return {"report_markdown": report_md, "result": res}

@app.get("/api/model/export")
def export_trained_model():
    ds_id = get_active_dataset_id()
    models_dir = os.path.join(DATA_DIR, ds_id, "models")
    os.makedirs(models_dir, exist_ok=True)
    weights_path = os.path.join(models_dir, "augmented_model_weights.pt")
    torchscript_path = os.path.join(models_dir, "augmented_torchscript.pt")

    if not os.path.exists(weights_path):
        benchmark_engine.run_experiment(epochs=6)

    inv = inspect_dataset_inventory(ds_id)
    classes = inv["classes"]

    infer_code = f'''import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

CLASSES = {classes}

class DefectConvNet(nn.Module):
    def __init__(self, num_classes={len(classes)}):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

def predict(img_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DefectConvNet().to(device)
    model.load_state_dict(torch.load("augmented_model_weights.pt", map_location=device))
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = probs.argmax().item()

    print(f"Predicted Class: {{CLASSES[idx]}} (Confidence: {{probs[idx]:.2%}})")
    for i, c in enumerate(CLASSES):
        print(f"  - {{c}}: {{probs[i]:.2%}}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py <image_path>")
    else:
        predict(sys.argv[1])
'''

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(weights_path):
            zf.write(weights_path, arcname="augmented_model_weights.pt")
        if os.path.exists(torchscript_path):
            zf.write(torchscript_path, arcname="augmented_torchscript.pt")
        zf.writestr("classes.json", json.dumps(classes, indent=2))
        zf.writestr("infer.py", infer_code)
        zf.writestr("README.md", f"# Trained Classifier Deployment Bundle\n\nDomain: {ds_id}\nClasses: {classes}\n\n## Usage:\n```bash\npython infer.py path/to/sample.jpg\n```\n")

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=pytorch_model_{ds_id.replace('/', '_')}.zip"}
    )

@app.get("/api/export/archive")
def export_archive(format: str = Query("raw", enum=["raw", "yolo", "coco"])):
    ds_id = get_active_dataset_id()
    real_dir, synth_dir = get_dataset_base_paths(ds_id)
    inv = inspect_dataset_inventory(ds_id)
    classes = inv["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = {
            "dataset_id": ds_id,
            "format": format,
            "classes": classes,
            "exported_at": os.environ.get("TIMESTAMP", "2026-10-02"),
            "items": []
        }
        
        coco_images = []
        coco_annotations = []
        ann_id = 1
        img_id = 1

        for category, base in [("real", real_dir), ("synthetic", synth_dir)]:
            for c in classes:
                cdir = os.path.join(base, c)
                if os.path.exists(cdir):
                    for f in sorted(os.listdir(cdir)):
                        if f.lower().endswith((".jpg", ".png", ".jpeg")):
                            src = os.path.join(cdir, f)
                            
                            if format == "yolo":
                                arc_name = f"images/{category}/{f}"
                                zf.write(src, arcname=arc_name)
                                # Generate normalized full-image bbox label for classification
                                label_name = f"labels/{category}/{os.path.splitext(f)[0]}.txt"
                                zf.writestr(label_name, f"{class_to_idx[c]} 0.5 0.5 1.0 1.0\n")
                            else:
                                arc_name = f"{ds_id}/{category}/{c}/{f}"
                                zf.write(src, arcname=arc_name)

                            if format == "coco":
                                coco_images.append({
                                    "id": img_id,
                                    "file_name": arc_name,
                                    "width": 256,
                                    "height": 256
                                })
                                coco_annotations.append({
                                    "id": ann_id,
                                    "image_id": img_id,
                                    "category_id": class_to_idx[c],
                                    "bbox": [0, 0, 256, 256],
                                    "area": 65536,
                                    "iscrowd": 0
                                })
                                ann_id += 1
                                img_id += 1

                            manifest["items"].append({
                                "file": arc_name,
                                "class": c,
                                "split": category
                            })

        if format == "yolo":
            yolo_yaml = f"names:\n" + "\n".join([f"  {i}: {c}" for i, c in enumerate(classes)]) + f"\nnc: {len(classes)}\ntrain: images/synthetic\nval: images/real\n"
            zf.writestr("data.yaml", yolo_yaml)
        elif format == "coco":
            coco_dict = {
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": [{"id": i, "name": c} for i, c in enumerate(classes)]
            }
            zf.writestr("coco_annotations.json", json.dumps(coco_dict, indent=2))
            
        zf.writestr(f"{ds_id}/manifest.json", json.dumps(manifest, indent=2))
        
    zip_buffer.seek(0)
    safe_name = ds_id.replace("/", "_")
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=dataset_{safe_name}_{format}.zip"}
    )

# Static file mounts
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
