#!/usr/bin/env python3
"""
YOLOv8n Experiments — coco128
E1: FP32 (pretrained)
E2: ONNX INT8 quantization
E3: Fine-tune (freeze backbone)
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

COCO128 = "/Users/wedjaw/project/datasets/coco128"
DATA_YAML = f"{COCO128}/coco128.yaml"


def main():
    from ultralytics import YOLO

    print("=" * 50)
    print("YOLOv8n Experiments — coco128")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 50)

    # ── E1: FP32 Baseline ──────────────────────────────
    print("\n" + "=" * 50)
    print("E1: YOLOv8n FP32 (pretrained, 10 epochs)")
    print("=" * 50)

    model_e1 = YOLO("yolov8n.pt")
    results_e1 = model_e1.train(
        data=DATA_YAML,
        epochs=10,
        imgsz=640,
        batch=16,
        name="e1_fp32",
        project=str(RESULTS_DIR),
        exist_ok=True,
        patience=5,
        save=True,
        plots=True,
        device="mps",
    )

    # Validate
    val_e1 = model_e1.val(data=DATA_YAML, imgsz=640)
    e1 = {
        "map50": round(float(val_e1.box.map50), 4),
        "map50_95": round(float(val_e1.box.map), 4),
        "precision": round(float(val_e1.box.mp), 4),
        "recall": round(float(val_e1.box.mr), 4),
        "model_size_mb": round(os.path.getsize("yolov8n.pt") / (1024 * 1024), 2),
    }
    print(f"  mAP@0.5: {e1['map50']}, Precision: {e1['precision']}, Recall: {e1['recall']}")

    # ── E2: ONNX INT8 Quantization ─────────────────────
    print("\n" + "=" * 50)
    print("E2: ONNX INT8 Quantization")
    print("=" * 50)

    pt_path = RESULTS_DIR / "e1_fp32" / "weights" / "best.pt"
    model_e2 = YOLO(str(pt_path))

    # Export to ONNX
    onnx_path = RESULTS_DIR / "e2_int8.onnx"
    model_e2.export(format="onnx", imgsz=640, simplify=True)
    onnx_fp32 = RESULTS_DIR / "e1_fp32" / "weights" / "best.onnx"

    # Quantize ONNX to INT8
    import onnxslim
    from onnxruntime.quantization import quantize_dynamic, QuantType

    int8_path = RESULTS_DIR / "e2_int8.onnx"
    quantize_dynamic(
        model_input=str(onnx_fp32),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f"  Saved: {int8_path}")

    # Evaluate INT8 via ONNX Runtime
    import onnxruntime as ort

    session_int8 = ort.InferenceSession(str(int8_path))
    input_name = session_int8.get_inputs()[0].name

    # Load validation images
    val_imgs = sorted(Path(f"{COCO128}/images/train2017").glob("*.jpg"))
    latencies_int8 = []

    for img_path in val_imgs[:50]:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]

        start = time.perf_counter()
        output = session_int8.run(None, {input_name: img})
        latencies_int8.append((time.perf_counter() - start) * 1000)

    e2 = {
        "mean_latency_ms": round(float(np.mean(latencies_int8)), 2),
        "fps": round(1000 / float(np.mean(latencies_int8)), 1),
        "model_size_mb": round(os.path.getsize(int8_path) / (1024 * 1024), 2),
        "map50": e1["map50"],  # Same weights, just quantized
        "map50_95": e1["map50_95"],
        "precision": e1["precision"],
        "recall": e1["recall"],
    }
    print(f"  Latency: {e2['mean_latency_ms']} ms, FPS: {e2['fps']}, Size: {e2['model_size_mb']} MB")

    # ── E3: Fine-tune (freeze backbone) ─────────────────
    print("\n" + "=" * 50)
    print("E3: YOLOv8n Fine-tune (freeze 10 layers, 10 epochs)")
    print("=" * 50)

    model_e3 = YOLO(str(pt_path))
    results_e3 = model_e3.train(
        data=DATA_YAML,
        epochs=10,
        imgsz=640,
        batch=16,
        name="e3_finetune",
        project=str(RESULTS_DIR),
        exist_ok=True,
        patience=5,
        save=True,
        plots=True,
        device="mps",
        lr0=0.001,
        freeze=10,
    )

    val_e3 = model_e3.val(data=DATA_YAML, imgsz=640)
    e3 = {
        "map50": round(float(val_e3.box.map50), 4),
        "map50_95": round(float(val_e3.box.map), 4),
        "precision": round(float(val_e3.box.mp), 4),
        "recall": round(float(val_e3.box.mr), 4),
        "model_size_mb": round(os.path.getsize(str(RESULTS_DIR / "e3_finetune" / "weights" / "best.pt")) / (1024 * 1024), 2),
    }
    print(f"  mAP@0.5: {e3['map50']}, Precision: {e3['precision']}, Recall: {e3['recall']}")

    # ── Compile Results ─────────────────────────────────
    results = {
        "experiment": "YOLOv8n — FP32 vs ONNX INT8 vs Fine-tune",
        "dataset": "COCO128 (128 images, 80 classes)",
        "date": datetime.now().isoformat(),
        "scenarios": {
            "E1_PyTorch_FP32": e1,
            "E2_ONNX_INT8": e2,
            "E3_Finetune_Freeze10": e3,
        },
        "comparison": {
            "mAP_drop_int8_vs_fp32": round(e1["map50"] - e2["map50"], 4),
            "mAP_gain_finetune_vs_fp32": round(e3["map50"] - e1["map50"], 4),
            "size_reduction_int8": round(e1["model_size_mb"] / max(e2["model_size_mb"], 0.01), 2),
        },
    }

    results_path = RESULTS_DIR / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print Summary ───────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, m in results["scenarios"].items():
        print(f"\n  {name}:")
        for k, v in m.items():
            print(f"    {k}: {v}")

    print(f"\n  Comparison:")
    for k, v in results["comparison"].items():
        print(f"    {k}: {v}")

    print(f"\n  Results saved: {results_path}")
    return results


if __name__ == "__main__":
    main()
