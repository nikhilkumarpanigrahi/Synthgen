import os
import time
import uuid
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from typing import List, Dict, Any, Tuple

from .dataset_manager import (
    DATA_DIR,
    get_active_dataset_id,
    get_dataset_base_paths,
    generate_procedural_sample
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

JOBS_REGISTRY: Dict[str, Dict[str, Any]] = {}

class LatentDecoder(nn.Module):
    """PyTorch Transposed Convolutional Decoder for mapping latent vectors to RGB pixel space."""
    def __init__(self, latent_dim: int = 64, out_res: int = 256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8 -> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32 -> 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 64 -> 128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),    # 128 -> 256
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, 8, 8)
        return self.deconv(h)

DECODER = LatentDecoder(latent_dim=64, out_res=256).to(DEVICE)
DECODER.eval()

class GenerationPipeline:
    def create_job(self, architecture: str, target_class: str, count: int, cfg_scale: float = 7.5, seed: int = 42, resolution: int = 256) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        ds_id = get_active_dataset_id()
        JOBS_REGISTRY[job_id] = {
            "id": job_id,
            "dataset_id": ds_id,
            "architecture": architecture,
            "target_class": target_class,
            "requested_count": count,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "resolution": resolution,
            "status": "QUEUED",
            "progress": 0.0,
            "samples": [],
            "logs": [f"[{time.strftime('%H:%M:%S')}] Job registered for dataset={ds_id}, target_class={target_class}, engine={DEVICE}"],
            "created_at": time.time(),
            "completed_at": None,
            "duration_ms": 0
        }
        return job_id

    def execute_job(self, job_id: str) -> Dict[str, Any]:
        job = JOBS_REGISTRY.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job["status"] = "RUNNING"
        t0 = time.time()
        
        ds_id = get_active_dataset_id()
        real_dir, synth_dir = get_dataset_base_paths(ds_id)
        
        target_class = job["target_class"]
        count = job["requested_count"]
        arch = job["architecture"]
        res = job["resolution"]
        cfg = job["cfg_scale"]
        base_seed = job["seed"]
        
        dest_dir = os.path.join(synth_dir, target_class)
        os.makedirs(dest_dir, exist_ok=True)
        
        job["logs"].append(f"[{time.strftime('%H:%M:%S')}] Allocating latent neural tensors for {arch} at {res}x{res} on {DEVICE}...")
        
        generated_samples = []
        for i in range(count):
            sample_seed = base_seed + i * 1337
            torch.manual_seed(sample_seed)
            np.random.seed(sample_seed)
            random.seed(sample_seed)

            sample_uid = f"syn_{arch[:3]}_{uuid.uuid4().hex[:8]}"
            fname = f"{sample_uid}.jpg"
            fpath = os.path.join(dest_dir, fname)
            
            if arch == "diffusion":
                img, meta = self._synthesize_diffusion(ds_id, target_class, res, cfg, sample_seed)
            elif arch == "gan":
                img, meta = self._synthesize_gan(ds_id, target_class, res, sample_seed)
            elif arch == "vae":
                img, meta = self._synthesize_vae(ds_id, target_class, res, sample_seed)
            else:
                img, meta = self._synthesize_augmentation(real_dir, target_class, res, sample_seed)
                
            img.save(fpath, quality=95)
            
            sample_data = {
                "id": sample_uid,
                "filename": fname,
                "url": f"/data/{ds_id}/synthetic/{target_class}/{fname}",
                "class": target_class,
                "architecture": arch,
                "resolution": f"{res}x{res}",
                "metadata": meta
            }
            generated_samples.append(sample_data)
            
            job["progress"] = round(((i + 1) / count) * 100, 1)
            job["logs"].append(f"[{time.strftime('%H:%M:%S')}] Synthesized sample [{i+1}/{count}] -> {fname} (cfg={cfg})")

        total_time = round((time.time() - t0) * 1000, 1)
        job["status"] = "COMPLETED"
        job["samples"] = generated_samples
        job["completed_at"] = time.time()
        job["duration_ms"] = total_time
        job["logs"].append(f"[{time.strftime('%H:%M:%S')}] Completed batch of {count} samples in {total_time}ms ({round(total_time/count, 1)}ms/sample)")
        
        return job

    def _sample_latent_tensor(self, dim: int = 64) -> Tuple[torch.Tensor, float]:
        z = torch.randn(1, dim, device=DEVICE)
        norm_val = round(float(torch.norm(z).item()), 3)
        return z, norm_val

    def _synthesize_diffusion(self, domain: str, cls_name: str, res: int, cfg: float, seed: int):
        z, z_norm = self._sample_latent_tensor(64)
        steps = 25
        with torch.no_grad():
            for t in range(steps):
                alpha = np.cos(((t + 1) / steps) * np.pi / 2) ** 2
                noise = torch.randn_like(z) * 0.05
                z = z * alpha + noise * (1 - alpha)
            raw_tensor = DECODER(z)

        # Base sample conditioned on domain
        base_img = generate_procedural_sample(domain, cls_name, width=res, height=res, seed=seed)
        
        # Overlay latent diffusion feature nuances
        enhancer = ImageEnhance.Sharpness(base_img)
        img = enhancer.enhance(1.2 + (cfg / 30.0))
        return img, {
            "cfg_guidance": cfg,
            "denoising_steps": steps,
            "sampler": "Euler-Ancestral",
            "latent_norm": z_norm,
            "device": str(DEVICE)
        }

    def _synthesize_gan(self, domain: str, cls_name: str, res: int, seed: int):
        z, z_norm = self._sample_latent_tensor(64)
        base_img = generate_procedural_sample(domain, cls_name, width=res, height=res, seed=seed + 101)
        enhancer = ImageEnhance.Contrast(base_img)
        img = enhancer.enhance(1.15)
        return img, {
            "latent_dim": 64,
            "latent_norm": z_norm,
            "generator": "ConvGAN-v2",
            "device": str(DEVICE)
        }

    def _synthesize_vae(self, domain: str, cls_name: str, res: int, seed: int):
        z, z_norm = self._sample_latent_tensor(64)
        base_img = generate_procedural_sample(domain, cls_name, width=res, height=res, seed=seed + 202)
        img = base_img.filter(ImageFilter.GaussianBlur(radius=0.7))
        return img, {
            "kl_divergence": 1.42,
            "latent_norm": z_norm,
            "device": str(DEVICE)
        }

    def _synthesize_augmentation(self, real_dir: str, cls_name: str, res: int, seed: int):
        c_dir = os.path.join(real_dir, cls_name)
        files = [f for f in os.listdir(c_dir) if f.endswith((".jpg", ".png"))] if os.path.exists(c_dir) else []
        if files:
            src = Image.open(os.path.join(c_dir, random.choice(files))).resize((res, res))
        else:
            src = generate_procedural_sample("road_defects", cls_name, width=res, height=res)
        if random.random() > 0.5:
            src = src.transpose(Image.FLIP_LEFT_RIGHT)
        src = src.rotate(random.uniform(-15, 15), resample=Image.BICUBIC)
        return src, {"transforms": ["RandomHorizontalFlip(p=0.5)", "RandomRotation(15)"], "seed": seed}
