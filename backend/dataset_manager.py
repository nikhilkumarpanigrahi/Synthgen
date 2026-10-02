import os
import shutil
import uuid
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REAL_DIR = os.path.join(DATA_DIR, "real")
SYNTH_DIR = os.path.join(DATA_DIR, "synthetic")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")

CLASSES = ["pothole", "surface_crack", "normal_road"]

def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(REAL_DIR, c), exist_ok=True)
        os.makedirs(os.path.join(SYNTH_DIR, c), exist_ok=True)

def generate_road_texture(width=256, height=256, base_val=70, noise_std=18):
    noise = np.random.normal(loc=base_val, scale=noise_std, size=(height, width, 3))
    # Aggregate stone/asphalt texture
    speckles = np.random.rand(height, width) > 0.94
    noise[speckles] = np.clip(noise[speckles] + np.random.randint(20, 45, size=noise[speckles].shape), 0, 255)
    img = Image.fromarray(np.uint8(np.clip(noise, 0, 255)))
    return img

def create_sample_pothole(seed=None):
    if seed: np.random.seed(seed)
    from PIL import ImageDraw, ImageFilter
    img = generate_road_texture(256, 256, base_val=68, noise_std=16)
    draw = ImageDraw.Draw(img)
    cx, cy = np.random.randint(80, 176), np.random.randint(80, 176)
    rx, ry = np.random.randint(35, 65), np.random.randint(25, 45)
    
    # Outer cavity
    pts = []
    for deg in range(0, 360, 20):
        rad = np.radians(deg)
        r = (rx if deg % 40 == 0 else rx * 0.85) + np.random.randint(-6, 6)
        pts.append((cx + r * np.cos(rad), cy + (r * 0.7) * np.sin(rad)))
    draw.polygon(pts, fill=(24, 22, 20))
    # Inner shadow
    inner = [(cx + (x - cx) * 0.65, cy + (y - cy) * 0.65) for x, y in pts]
    draw.polygon(inner, fill=(12, 10, 10))
    return img.filter(ImageFilter.SMOOTH)

def create_sample_crack(seed=None):
    if seed: np.random.seed(seed)
    from PIL import ImageDraw, ImageFilter
    img = generate_road_texture(256, 256, base_val=74, noise_std=18)
    draw = ImageDraw.Draw(img)
    x = np.random.randint(30, 226)
    y = 0
    while y < 256:
        nx = x + np.random.randint(-12, 12)
        ny = y + np.random.randint(10, 22)
        draw.line([(x, y), (nx, ny)], fill=(18, 18, 18), width=np.random.randint(2, 3))
        x, y = nx, ny
    return img.filter(ImageFilter.SMOOTH)

def create_sample_normal(seed=None):
    if seed: np.random.seed(seed)
    from PIL import ImageDraw
    img = generate_road_texture(256, 256, base_val=80, noise_std=14)
    draw = ImageDraw.Draw(img)
    if np.random.rand() > 0.4:
        draw.line([(128, 0), (128, 256)], fill=(210, 210, 205), width=5)
    return img

def init_default_dataset():
    ensure_storage()
    counts = {"pothole": 24, "surface_crack": 32, "normal_road": 60}
    for cls_name, target in counts.items():
        cls_dir = os.path.join(REAL_DIR, cls_name)
        curr = [f for f in os.listdir(cls_dir) if f.endswith((".jpg", ".png"))]
        if len(curr) < target:
            for i in range(len(curr), target):
                fname = f"real_{cls_name}_{i+1:03d}.jpg"
                p = os.path.join(cls_dir, fname)
                if cls_name == "pothole":
                    im = create_sample_pothole(seed=5000 + i)
                elif cls_name == "surface_crack":
                    im = create_sample_crack(seed=6000 + i)
                else:
                    im = create_sample_normal(seed=7000 + i)
                im.save(p, quality=92)
    return get_dataset_inventory()

def get_dataset_inventory() -> Dict[str, Any]:
    ensure_storage()
    real_counts = {}
    synth_counts = {}
    
    for c in CLASSES:
        r_files = [f for f in os.listdir(os.path.join(REAL_DIR, c)) if f.endswith((".jpg", ".png"))]
        s_files = [f for f in os.listdir(os.path.join(SYNTH_DIR, c)) if f.endswith((".jpg", ".png"))]
        real_counts[c] = len(r_files)
        synth_counts[c] = len(s_files)
        
    total_real = sum(real_counts.values())
    total_synth = sum(synth_counts.values())
    
    imbalance_ratio = round(max(real_counts.values()) / max(1, min(real_counts.values())), 2)
    
    return {
        "classes": CLASSES,
        "real": {
            "total": total_real,
            "counts": real_counts,
            "imbalance_ratio": imbalance_ratio
        },
        "synthetic": {
            "total": total_synth,
            "counts": synth_counts
        },
        "combined_total": total_real + total_synth,
        "storage_path": DATA_DIR
    }

def save_uploaded_image(file_bytes: bytes, filename: str, target_class: str) -> Dict[str, Any]:
    ensure_storage()
    if target_class not in CLASSES:
        target_class = "pothole"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        ext = ".jpg"
    uid = uuid.uuid4().hex[:8]
    saved_name = f"user_{target_class}_{uid}{ext}"
    dest = os.path.join(REAL_DIR, target_class, saved_name)
    
    with open(dest, "wb") as f:
        f.write(file_bytes)
        
    return {
        "id": uid,
        "filename": saved_name,
        "class": target_class,
        "url": f"/data/real/{target_class}/{saved_name}"
    }

def get_samples_list(category: str = "real", target_class: Optional[str] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    ensure_storage()
    base_dir = REAL_DIR if category == "real" else SYNTH_DIR
    classes_to_scan = [target_class] if target_class and target_class in CLASSES else CLASSES
    
    items = []
    for c in classes_to_scan:
        c_dir = os.path.join(base_dir, c)
        if os.path.exists(c_dir):
            files = sorted([f for f in os.listdir(c_dir) if f.endswith((".jpg", ".png"))], reverse=True)
            for f in files:
                fpath = os.path.join(c_dir, f)
                stat = os.stat(fpath)
                items.append({
                    "id": f.split(".")[0],
                    "filename": f,
                    "class": c,
                    "category": category,
                    "size_bytes": stat.st_size,
                    "url": f"/data/{category}/{c}/{f}",
                    "created_at": int(stat.st_mtime)
                })
                
    total = len(items)
    paginated = items[offset:offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": paginated
    }
