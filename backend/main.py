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

@app.get("/api/export/archive")
def export_archive(format: str = Query("raw", enum=["raw", "yolo", "coco"])):
    ds_id = get_active_dataset_id()
    real_dir, synth_dir = get_dataset_base_paths(ds_id)
    inv = inspect_dataset_inventory(ds_id)
    classes = inv["classes"]

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = {
            "dataset_id": ds_id,
            "format": format,
            "classes": classes,
            "exported_at": os.environ.get("TIMESTAMP", "2026-10-02"),
            "items": []
        }
        
        for category, base in [("real", real_dir), ("synthetic", synth_dir)]:
            for c in classes:
                cdir = os.path.join(base, c)
                if os.path.exists(cdir):
                    for f in sorted(os.listdir(cdir)):
                        if f.lower().endswith((".jpg", ".png", ".jpeg")):
                            src = os.path.join(cdir, f)
                            arc_name = f"{ds_id}/{category}/{c}/{f}"
                            zf.write(src, arcname=arc_name)
                            manifest["items"].append({
                                "file": arc_name,
                                "class": c,
                                "split": category
                            })
                            
        zf.writestr(f"{ds_id}/manifest.json", json.dumps(manifest, indent=2))
        
    zip_buffer.seek(0)
    safe_name = ds_id.replace("/", "_")
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=synthetic_augmented_{safe_name}.zip"}
    )

# Static file mounts
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
