import os
import time
import random
import numpy as np
from typing import Dict, Any, List, Tuple
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from .dataset_manager import REAL_DIR, SYNTH_DIR, CLASSES

# Device selection: Apple Silicon MPS if available, else CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

class RoadDefectDataset(Dataset):
    def __init__(self, file_paths: List[Tuple[str, int]], transform=None):
        self.samples = file_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
        tensor_img = self.transform(rgb_img)
        return tensor_img, label

class DefectConvNet(nn.Module):
    """Convolutional Neural Network for multi-class road defect classification."""
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 64 -> 32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 32 -> 16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)) # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        flat = feat.view(feat.size(0), -1)
        return self.classifier(flat)

class DownstreamClassifierBenchmark:
    """
    Live PyTorch Training & Validation Benchmark:
    - Trains Model A on real dataset only.
    - Trains Model B on real + synthetic generated dataset.
    - Evaluates both models on identical held-out real validation images.
    - Computes real loss convergence, validation accuracy, precision, recall, F1, and confusion matrix.
    """

    def _collect_data_splits(self) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
        real_train = []
        real_val = []
        synth_train = []

        # Real dataset split: 75% train, 25% validation per class (deterministic seed)
        for c in CLASSES:
            c_dir = os.path.join(REAL_DIR, c)
            if os.path.exists(c_dir):
                files = sorted([os.path.join(c_dir, f) for f in os.listdir(c_dir) if f.endswith((".jpg", ".png"))])
                lbl = CLASS_TO_IDX[c]
                
                rng = random.Random(42)
                shuffled = files.copy()
                rng.shuffle(shuffled)
                
                split_idx = max(1, int(len(shuffled) * 0.75))
                for p in shuffled[:split_idx]:
                    real_train.append((p, lbl))
                for p in shuffled[split_idx:]:
                    real_val.append((p, lbl))

        # Synthetic samples (all used for training augmentation)
        for c in CLASSES:
            s_dir = os.path.join(SYNTH_DIR, c)
            if os.path.exists(s_dir):
                files = sorted([os.path.join(s_dir, f) for f in os.listdir(s_dir) if f.endswith((".jpg", ".png"))])
                lbl = CLASS_TO_IDX[c]
                for p in files:
                    synth_train.append((p, lbl))

        return real_train, synth_train, real_val

    def run_experiment(self, backbone: str = "DefectConvNet-V2", epochs: int = 12, learning_rate: float = 0.002) -> Dict[str, Any]:
        t0 = time.time()
        
        real_train_files, synth_train_files, val_files = self._collect_data_splits()
        
        # Datasets
        ds_baseline_train = RoadDefectDataset(real_train_files)
        ds_augmented_train = RoadDefectDataset(real_train_files + synth_train_files)
        ds_val = RoadDefectDataset(val_files)

        val_loader = DataLoader(ds_val, batch_size=16, shuffle=False)

        # 1. Train Model A (Baseline: Real Data Only)
        model_a = DefectConvNet(num_classes=len(CLASSES)).to(DEVICE)
        loader_a = DataLoader(ds_baseline_train, batch_size=16, shuffle=True)
        res_a = self._train_and_evaluate(model_a, loader_a, val_loader, epochs, learning_rate, seed=101)

        # 2. Train Model B (Augmented: Real + Synthetic Data)
        model_b = DefectConvNet(num_classes=len(CLASSES)).to(DEVICE)
        loader_b = DataLoader(ds_augmented_train, batch_size=16, shuffle=True)
        res_b = self._train_and_evaluate(model_b, loader_b, val_loader, epochs, learning_rate, seed=202)

        total_duration = round(time.time() - t0, 2)

        # Calculate live deltas
        acc_delta = round(res_b["accuracy"] - res_a["accuracy"], 1)
        f1_delta = round(res_b["macro_f1"] - res_a["macro_f1"], 1)
        pothole_idx = CLASS_TO_IDX["pothole"]
        pothole_rec_delta = round(res_b["per_class_recall"][pothole_idx] - res_a["per_class_recall"][pothole_idx], 1)

        return {
            "config": {
                "backbone": backbone,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "device": str(DEVICE),
                "optimizer": "AdamW (weight_decay=0.01)",
                "loss_function": "CrossEntropyLoss",
                "training_duration_sec": total_duration
            },
            "dataset_sizes": {
                "baseline_train": len(ds_baseline_train),
                "synthetic_augmented": len(ds_augmented_train),
                "validation_set": len(ds_val)
            },
            "summary_comparison": {
                "accuracy": {
                    "baseline": res_a["accuracy"],
                    "augmented": res_b["accuracy"],
                    "delta": f"{'+' if acc_delta >= 0 else ''}{acc_delta}%"
                },
                "macro_f1": {
                    "baseline": res_a["macro_f1"],
                    "augmented": res_b["macro_f1"],
                    "delta": f"{'+' if f1_delta >= 0 else ''}{f1_delta}%"
                },
                "rare_defect_recall": {
                    "baseline": res_a["per_class_recall"][pothole_idx],
                    "augmented": res_b["per_class_recall"][pothole_idx],
                    "delta": f"{'+' if pothole_rec_delta >= 0 else ''}{pothole_rec_delta}%"
                }
            },
            "classification_reports": {
                "baseline": res_a["classification_report"],
                "augmented": res_b["classification_report"]
            },
            "confusion_matrices": {
                "baseline": {"labels": CLASSES, "matrix": res_a["confusion_matrix"]},
                "augmented": {"labels": CLASSES, "matrix": res_b["confusion_matrix"]}
            },
            "training_curves": {
                "epochs": list(range(1, epochs + 1)),
                "loss_baseline": res_a["epoch_losses"],
                "loss_augmented": res_b["epoch_losses"],
                "val_acc_baseline": res_a["epoch_val_accs"],
                "val_acc_augmented": res_b["epoch_val_accs"]
            }
        }

    def _train_and_evaluate(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

        epoch_losses = []
        epoch_val_accs = []

        for ep in range(epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

            ep_loss = round(running_loss / max(1, total_samples), 4)
            epoch_losses.append(ep_loss)

            # Evaluate on val set at this epoch
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    preds = model(inputs).argmax(dim=1)
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.size(0)
            val_acc = round((val_correct / max(1, val_total)) * 100, 1)
            epoch_val_accs.append(val_acc)

        # Final evaluation
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                preds = model(inputs).argmax(dim=1)
                y_true.extend(targets.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        overall_acc = round(accuracy_score(y_true, y_pred) * 100, 1)
        prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

        macro_f1 = round(float(np.mean(f1)) * 100, 1)
        
        report = []
        per_class_rec = []
        for i, c in enumerate(CLASSES):
            report.append({
                "class": c,
                "precision": round(float(prec[i]), 2),
                "recall": round(float(rec[i]), 2),
                "f1_score": round(float(f1[i]), 2),
                "support": int(support[i])
            })
            per_class_rec.append(round(float(rec[i]) * 100, 1))

        return {
            "accuracy": overall_acc,
            "macro_f1": macro_f1,
            "per_class_recall": per_class_rec,
            "classification_report": report,
            "confusion_matrix": cm,
            "epoch_losses": epoch_losses,
            "epoch_val_accs": epoch_val_accs
        }
