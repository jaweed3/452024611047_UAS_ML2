#!/usr/bin/env python3
"""
Train YOLOv8n on Pascal VOC person detection
"""

import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_YAML = PROJECT_ROOT / "voc_person.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

def train():
    from ultralytics import YOLO
    
    print("=" * 50)
    print("YOLOv8n Training — Pascal VOC Person")
    print("=" * 50)
    
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load pretrained YOLOv8n
    model = YOLO("yolov8n.pt")
    
    # Train
    print("\n[Training] YOLOv8n (10 epochs, imgsz=640)")
    results = model.train(
        data=str(DATASET_YAML),
        epochs=10,
        imgsz=640,
        batch=16,
        name="yolov8n_voc_person",
        project=str(MODELS_DIR),
        exist_ok=True,
        patience=5,
        save=True,
        plots=True,
        device="mps",  # Mac M4 GPU
    )
    
    print(f"\nTraining complete. Results: {MODELS_DIR / 'yolov8n_voc_person'}")
    return results

if __name__ == "__main__":
    train()
