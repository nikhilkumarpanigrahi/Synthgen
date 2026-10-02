import os
import shutil
import zipfile
import uuid
import json
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")

ACTIVE_CONFIG_PATH = os.path.join(DATA_DIR, "active_dataset.json")
REAL_DIR = os.path.join(DATA_DIR, "road_defects", "real")
SYNTH_DIR = os.path.join(DATA_DIR, "road_defects", "synthetic")
CLASSES = ["pothole", "surface_crack", "normal_road"]

DOMAINS = {
    "road_defects": {
        "name": "Civil Infrastructure: Road Surface Defects",
        "description": "Asphalt road conditions with extreme scarcity of critical pothole safety hazards.",
        "classes": ["pothole", "surface_crack", "normal_road"],
        "rare_class": "pothole",
        "counts": {"pothole": 16, "surface_crack": 32, "normal_road": 64}
    },
    "medical_imaging": {
        "name": "Dermatology: Pigmented Skin Lesions",
        "description": "Histopathological dermatoscopy images with rare malignant melanoma cases.",
        "classes": ["malignant_melanoma", "keratosis", "benign_nevus"],
        "rare_class": "malignant_melanoma",
        "counts": {"malignant_melanoma": 12, "keratosis": 30, "benign_nevus": 72}
    },
    "industrial_defects": {
        "name": "Manufacturing: Steel Surface Anomalies",
        "description": "Automated quality control inspection with critical rare micro-crack defects.",
        "classes": ["micro_fracture", "welding_void", "normal_surface"],
        "rare_class": "micro_fracture",
        "counts": {"micro_fracture": 14, "welding_void": 32, "normal_surface": 60}
    }
}

def get_active_dataset_id() -> str:
    if os.path.exists(ACTIVE_CONFIG_PATH):
        try:
            with open(ACTIVE_CONFIG_PATH, "r") as f:
                data = json.load(f)
                return data.get("active_dataset", "road_defects")
        except Exception:
            pass
    return "road_defects"

def set_active_dataset_id(dataset_id: str):
    with open(ACTIVE_CONFIG_PATH, "w") as f:
        json.dump({"active_dataset": dataset_id}, f)

def get_dataset_base_paths(dataset_id: Optional[str] = None) -> Tuple[str, str]:
    ds_id = dataset_id or get_active_dataset_id()
    ds_dir = os.path.join(DATA_DIR, ds_id)
    real_dir = os.path.join(ds_dir, "real")
    synth_dir = os.path.join(ds_dir, "synthetic")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(synth_dir, exist_ok=True)
    return real_dir, synth_dir

# Procedural Image Generators for Domains
def generate_procedural_sample(domain: str, cls_name: str, width: int = 256, height: int = 256, seed: int = None) -> Image.Image:
    if seed:
        np.random.seed(seed)
    
    if domain == "medical_imaging":
        # Skin dermoscopy procedural rendering
        base_skin = np.random.normal(loc=190, scale=12, size=(height, width, 3))
        base_skin[:, :, 1] -= 30 # pinkish/peach skin tone
        base_skin[:, :, 2] -= 45
        img = Image.fromarray(np.uint8(np.clip(base_skin, 0, 255))).filter(ImageFilter.GaussianBlur(1.2))
        draw = ImageDraw.Draw(img)
        cx, cy = width // 2, height // 2
        
        if cls_name == "malignant_melanoma":
            # Asymmetrical, irregular dark lesion (ABCD criteria for rare melanoma)
            pts = []
            steps = 24
            rad = np.random.randint(45, 75)
            for k in range(steps):
                th = (2 * np.pi / steps) * k
                r = rad * (1.0 + np.random.normal(0, 0.35))
                pts.append((cx + r * np.cos(th), cy + r * np.sin(th)))
            draw.polygon(pts, fill=(45, 25, 20)) # Dark black/brown
            # Inner irregular pigmentation
            inner = [(cx + (x - cx) * 0.6, cy + (y - cy) * 0.6) for x, y in pts]
            draw.polygon(inner, fill=(20, 10, 10))
        elif cls_name == "keratosis":
            # Rough, crusty brownish patch
            rad = np.random.randint(35, 55)
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(130, 85, 60))
            for _ in range(30):
                rx = cx + np.random.randint(-rad, rad)
                ry = cy + np.random.randint(-rad, rad)
                draw.point((rx, ry), fill=(70, 40, 25))
        else: # benign_nevus
            # Symmetrical, even round brown mole
            rad = np.random.randint(30, 45)
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(100, 60, 40))
        return img.filter(ImageFilter.SMOOTH)

    elif domain == "industrial_defects":
        # Cold rolled steel surface
        base_metal = np.random.normal(loc=130, scale=10, size=(height, width, 3))
        img = Image.fromarray(np.uint8(np.clip(base_metal, 0, 255)))
        draw = ImageDraw.Draw(img)
        
        if cls_name == "micro_fracture":
            # Sharp hairline defect
            x, y = np.random.randint(40, 216), 10
            for _ in range(14):
                nx = x + np.random.randint(-10, 10)
                ny = y + np.random.randint(14, 22)
                draw.line([(x, y), (nx, ny)], fill=(30, 30, 30), width=2)
                draw.line([(x+1, y), (nx+1, ny)], fill=(190, 190, 190), width=1)
                x, y = nx, ny
        elif cls_name == "welding_void":
            # Circular pitting/void defect
            for _ in range(np.random.randint(3, 7)):
                px = np.random.randint(50, 206)
                py = np.random.randint(50, 206)
                pr = np.random.randint(8, 20)
                draw.ellipse([px-pr, py-pr, px+pr, py+pr], fill=(35, 35, 35))
                draw.arc([px-pr-1, py-pr-1, px+pr+1, py+pr+1], 0, 360, fill=(180, 180, 180))
        else:
            # Clean rolled steel with subtle rolling grain
            for y in range(0, height, 16):
                draw.line([(0, y), (width, y)], fill=(125, 125, 125), width=1)
        return img.filter(ImageFilter.SMOOTH)

    else:
        # Default: Road defects (pothole, crack, normal)
        base = np.random.normal(loc=70, scale=16, size=(height, width, 3))
        img = Image.fromarray(np.uint8(np.clip(base, 0, 255)))
        draw = ImageDraw.Draw(img)
        if cls_name == "pothole":
            cx, cy = np.random.randint(80, 176), np.random.randint(80, 176)
            rad = np.random.randint(40, 65)
            pts = []
            for deg in range(0, 360, 20):
                r = rad * (1.0 + np.random.normal(0, 0.2))
                pts.append((cx + r * np.cos(np.radians(deg)), cy + (r * 0.7) * np.sin(np.radians(deg))))
            draw.polygon(pts, fill=(22, 20, 18))
            inner = [(cx + (x - cx) * 0.65, cy + (y - cy) * 0.65) for x, y in pts]
            draw.polygon(inner, fill=(12, 10, 10))
        elif cls_name == "surface_crack":
            x, y = np.random.randint(30, 226), 0
            while y < height:
                nx, ny = x + np.random.randint(-12, 12), y + np.random.randint(10, 22)
                draw.line([(x, y), (nx, ny)], fill=(18, 18, 18), width=2)
                x, y = nx, ny
        else:
            if np.random.rand() > 0.4:
                draw.line([(width//2, 0), (width//2, height)], fill=(210, 210, 205), width=5)
        return img.filter(ImageFilter.SMOOTH)

def initialize_domain_datasets():
    """Initializes standard benchmark datasets for multiple scientific and engineering domains."""
    for ds_id, meta in DOMAINS.items():
        real_dir, synth_dir = get_dataset_base_paths(ds_id)
        for cls_name in meta["classes"]:
            c_dir = os.path.join(real_dir, cls_name)
            os.makedirs(c_dir, exist_ok=True)
            target_count = meta["counts"][cls_name]
            existing = [f for f in os.listdir(c_dir) if f.endswith((".jpg", ".png"))]
            if len(existing) < target_count:
                for i in range(len(existing), target_count):
                    fname = f"real_{cls_name}_{i+1:03d}.jpg"
                    im = generate_procedural_sample(ds_id, cls_name, seed=1000 + i * 37)
                    im.save(os.path.join(c_dir, fname), quality=92)

def get_available_datasets() -> List[Dict[str, Any]]:
    datasets = []
    active_id = get_active_dataset_id()
    
    # 1. Built-in domains
    for ds_id, meta in DOMAINS.items():
        inv = inspect_dataset_inventory(ds_id)
        datasets.append({
            "id": ds_id,
            "name": meta["name"],
            "description": meta["description"],
            "classes": meta["classes"],
            "is_active": (ds_id == active_id),
            "is_custom": False,
            "inventory": inv
        })
        
    # 2. Custom uploaded datasets
    custom_dir = os.path.join(DATA_DIR, "custom")
    if os.path.exists(custom_dir):
        for item in os.listdir(custom_dir):
            p = os.path.join(custom_dir, item)
            if os.path.isdir(p):
                inv = inspect_dataset_inventory(f"custom/{item}")
                datasets.append({
                    "id": f"custom/{item}",
                    "name": f"Custom: {item.replace('_', ' ').title()}",
                    "description": "User-ingested custom classification dataset.",
                    "classes": inv["classes"],
                    "is_active": (f"custom/{item}" == active_id),
                    "is_custom": True,
                    "inventory": inv
                })
    return datasets

def inspect_dataset_inventory(dataset_id: Optional[str] = None) -> Dict[str, Any]:
    ds_id = dataset_id or get_active_dataset_id()
    real_dir, synth_dir = get_dataset_base_paths(ds_id)
    
    # Scan real classes
    real_counts = {}
    if os.path.exists(real_dir):
        for c in sorted(os.listdir(real_dir)):
            cp = os.path.join(real_dir, c)
            if os.path.isdir(cp):
                imgs = [f for f in os.listdir(cp) if f.endswith((".jpg", ".png", ".jpeg"))]
                real_counts[c] = len(imgs)
                
    # Scan synthetic counts
    synth_counts = {}
    if os.path.exists(synth_dir):
        for c in sorted(os.listdir(synth_dir)):
            cp = os.path.join(synth_dir, c)
            if os.path.isdir(cp):
                imgs = [f for f in os.listdir(cp) if f.endswith((".jpg", ".png", ".jpeg"))]
                synth_counts[c] = len(imgs)

    classes = list(real_counts.keys())
    if not classes:
        classes = ["class_a", "class_b"]
        real_counts = {"class_a": 0, "class_b": 0}

    # Automatically identify underrepresented / rare class
    sorted_by_count = sorted(real_counts.items(), key=lambda x: x[1])
    rare_class, min_count = sorted_by_count[0]
    max_count = max(real_counts.values()) if real_counts.values() else 1
    imbalance_ratio = round(max_count / max(1, min_count), 2)
    
    # Parity quota needed
    augmentation_quota = max(0, max_count - min_count)

    return {
        "dataset_id": ds_id,
        "classes": classes,
        "class_counts_real": real_counts,
        "class_counts_synthetic": synth_counts,
        "total_real": sum(real_counts.values()),
        "total_synthetic": sum(synth_counts.values()),
        "analysis": {
            "underrepresented_class": rare_class,
            "minority_count": min_count,
            "majority_count": max_count,
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": imbalance_ratio >= 1.5,
            "recommended_augmentation_quota": augmentation_quota,
            "severity": "CRITICAL" if imbalance_ratio >= 3.0 else "MODERATE" if imbalance_ratio >= 1.5 else "BALANCED"
        }
    }

def ingest_custom_zip(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Ingests and unzips a user dataset folder structure where subdirectories are class names."""
    dataset_name = os.path.splitext(filename)[0].lower().replace(" ", "_")
    custom_root = os.path.join(DATA_DIR, "custom", dataset_name)
    real_dir = os.path.join(custom_root, "real")
    synth_dir = os.path.join(custom_root, "synthetic")
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(synth_dir, exist_ok=True)
    
    temp_zip = os.path.join(UPLOADS_DIR, f"temp_{uuid.uuid4().hex[:6]}.zip")
    with open(temp_zip, "wb") as f:
        f.write(file_bytes)
        
    extracted_classes = set()
    with zipfile.ZipFile(temp_zip, "r") as zf:
        for member in zf.infolist():
            if not member.is_dir():
                parts = member.filename.strip("/").split("/")
                if len(parts) >= 2:
                    cls_name = parts[-2].lower().replace(" ", "_")
                    extracted_classes.add(cls_name)
                    c_dest = os.path.join(real_dir, cls_name)
                    os.makedirs(c_dest, exist_ok=True)
                    
                    fname = os.path.basename(member.filename)
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        with zf.open(member) as src, open(os.path.join(c_dest, fname), "wb") as dst:
                            shutil.copyfileobj(src, dst)
                            
    os.remove(temp_zip)
    
    ds_id = f"custom/{dataset_name}"
    set_active_dataset_id(ds_id)
    return inspect_dataset_inventory(ds_id)

def get_dataset_samples(category: str = "real", class_name: Optional[str] = None, limit: int = 40, offset: int = 0) -> Dict[str, Any]:
    ds_id = get_active_dataset_id()
    real_dir, synth_dir = get_dataset_base_paths(ds_id)
    base_dir = real_dir if category == "real" else synth_dir
    
    items = []
    if os.path.exists(base_dir):
        subdirs = [class_name] if class_name else sorted(os.listdir(base_dir))
        for c in subdirs:
            cdir = os.path.join(base_dir, c)
            if os.path.isdir(cdir):
                for f in sorted(os.listdir(cdir), reverse=True):
                    if f.endswith((".jpg", ".png", ".jpeg")):
                        fp = os.path.join(cdir, f)
                        st = os.stat(fp)
                        items.append({
                            "id": f.split(".")[0],
                            "filename": f,
                            "class": c,
                            "category": category,
                            "size_bytes": st.st_size,
                            "url": f"/data/{ds_id}/{category}/{c}/{f}"
                        })
                        
    total = len(items)
    return {
        "dataset_id": ds_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": items[offset:offset + limit]
    }
