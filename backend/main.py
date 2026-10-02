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
    init_default_dataset,
    get_dataset_inventory,
    get_samples_list,
    save_uploaded_image,
    DATA_DIR,
    REAL_DIR,
    SYNTH_DIR,
    CLASSES
)
from .generator import GenerationPipeline, JOBS_REGISTRY
from .evaluator import QualityAssuranceEngine
from .cnn_trainer import DownstreamClassifierBenchmark

app = FastAPI(
    title="Synthetix ML Platform",
    description="Enterprise Synthetic Data Generation and Quality Assurance Platform for Computer Vision",
    version="2.0.0"
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

class JobCreateRequest(BaseModel):
    architecture: str = "diffusion"  # diffusion, gan, vae, augmentation
    target_class: str = "pothole"
    count: int = 8
    cfg_scale: float = 7.5
    seed: int = 42
    resolution: int = 256

class BenchmarkRunRequest(BaseModel):
    backbone: str = "ResNet-18"
    epochs: int = 15
    learning_rate: float = 0.001

@app.on_event("startup")
def startup_event():
    init_default_dataset()

@app.get("/api/system/status")
def system_status():
    return {
        "status": "HEALTHY",
        "version": "2.0.0",
        "engine": "PyTorch 2.14 Cuda/MPS Accelerated",
        "storage": DATA_DIR
    }

@app.get("/api/dataset/inventory")
def dataset_inventory():
    return get_dataset_inventory()

@app.get("/api/dataset/samples")
def dataset_samples(
    category: str = Query("real", enum=["real", "synthetic"]),
    class_name: Optional[str] = None,
    limit: int = Query(40, le=200),
    offset: int = Query(0, ge=0)
):
    return get_samples_list(category=category, target_class=class_name, limit=limit, offset=offset)

@app.post("/api/dataset/upload")
async def upload_image(file: UploadFile = File(...), target_class: str = Form("pothole")):
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Max file size 10MB")
    res = save_uploaded_image(content, file.filename, target_class)
    return {"status": "SUCCESS", "uploaded": res}

@app.post("/api/jobs/create")
def create_job(req: JobCreateRequest):
    if req.target_class not in CLASSES:
        raise HTTPException(status_code=400, detail=f"Invalid class. Allowed: {CLASSES}")
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
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = {
            "format": format,
            "classes": CLASSES,
            "exported_at": os.environ.get("TIMESTAMP", "2026-10-02"),
            "items": []
        }
        
        for category in ["real", "synthetic"]:
            base = REAL_DIR if category == "real" else SYNTH_DIR
            for c in CLASSES:
                cdir = os.path.join(base, c)
                if os.path.exists(cdir):
                    for f in os.listdir(cdir):
                        if f.endswith((".jpg", ".png")):
                            src = os.path.join(cdir, f)
                            arc_name = f"dataset/{category}/{c}/{f}"
                            zf.write(src, arcname=arc_name)
                            manifest["items"].append({
                                "file": arc_name,
                                "class": c,
                                "split": category
                            })
                            
        zf.writestr("dataset/manifest.json", json.dumps(manifest, indent=2))
        
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=synthetix_dataset_{format}.zip"}
    )

# Static file serving
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
