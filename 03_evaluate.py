#!/usr/bin/env python3
"""
Evaluate YOLOv8n: FP32 (PyTorch) vs TFLite FP32 vs TFLite INT8
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── Evaluation Metrics ──────────────────────────────────
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def compute_map(predictions, ground_truths, iou_thresh=0.5):
    all_tp = []
    all_fp = []
    all_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred.get("boxes", [])
        pred_scores = pred.get("scores", [])
        gt_boxes = gt.get("boxes", [])
        
        matched_gt = set()
        
        for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
            all_scores.append(score)
            best_iou = 0
            best_gt = -1
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = compute_iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = j
            if best_iou >= iou_thresh and best_gt >= 0:
                all_tp.append(1)
                all_fp.append(0)
                matched_gt.add(best_gt)
            else:
                all_tp.append(0)
                all_fp.append(1)
        
        for _ in range(len(gt_boxes) - len(matched_gt)):
            all_tp.append(0)
            all_fp.append(1)
    
    if not all_tp:
        return {"mAP@0.5": 0, "precision": 0, "recall": 0}
    
    sorted_idx = np.argsort(all_scores)[::-1]
    tp_cum = np.cumsum([all_tp[i] for i in sorted_idx])
    fp_cum = np.cumsum([all_fp[i] for i in sorted_idx])
    
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / max(sum(1 for gt in ground_truths for _ in gt.get("boxes", [])), 1)
    
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        p_at_t = precision[recall >= t]
        if len(p_at_t) > 0:
            ap += np.max(p_at_t)
    ap /= 11
    
    return {
        "mAP@0.5": float(ap),
        "precision": float(precision[-1]) if len(precision) > 0 else 0,
        "recall": float(recall[-1]) if len(recall) > 0 else 0,
    }

# ── PyTorch Evaluation ─────────────────────────────────
def evaluate_pytorch(model_path, val_images, val_labels, imgsz=640):
    from ultralytics import YOLO
    import cv2
    
    model = YOLO(str(model_path))
    
    predictions = []
    latencies = []
    
    for img_path in tqdm(val_images[:100], desc="  PyTorch eval"):
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (imgsz, imgsz))
        
        start = time.perf_counter()
        results = model(img, verbose=False)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)
        
        pred_boxes = []
        pred_scores = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    pred_boxes.append(box.xyxy[0].cpu().numpy().tolist())
                    pred_scores.append(float(box.conf[0]))
        
        predictions.append({"boxes": pred_boxes, "scores": pred_scores})
    
    gt = load_ground_truth(val_labels[:100], imgsz)
    metrics = compute_map(predictions, gt)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 2)
    metrics["fps"] = round(1000 / metrics["mean_latency_ms"], 1)
    metrics["model_size_mb"] = round(os.path.getsize(model_path) / (1024 * 1024), 2)
    
    return metrics

# ── TFLite Evaluation ──────────────────────────────────
def evaluate_tflite(tflite_path, val_images, val_labels, imgsz=640):
    import tensorflow as tf
    import cv2
    
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_dtype = input_details[0]["dtype"]
    input_quant = input_details[0].get("quantization", (1.0, 0))
    
    predictions = []
    latencies = []
    
    for img_path in tqdm(val_images[:100], desc="  TFLite eval"):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (imgsz, imgsz))
        img = img.astype(np.float32) / 255.0
        
        if input_dtype == np.uint8:
            inp = (img * 255).astype(np.uint8)
        elif input_dtype == np.int8:
            in_scale, in_zp = input_quant
            if in_scale is None or in_scale == 0:
                in_scale = 1.0
            if in_zp is None:
                in_zp = 0
            inp = np.clip(img / in_scale + in_zp, -128, 127).astype(np.int8)
        else:
            inp = img.astype(np.float32)
        
        interpreter.set_tensor(input_details[0]["index"], inp[np.newaxis, ...])
        
        start = time.perf_counter()
        interpreter.invoke()
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)
        
        output = interpreter.get_tensor(output_details[0]["index"])
        output_dtype = output_details[0]["dtype"]
        output_quant = output_details[0].get("quantization", (0, 0))
        
        if output_dtype in (np.uint8, np.int8):
            scale, zp = output_quant
            if scale is not None and zp is not None:
                output = (output.astype(np.float32) - zp) * scale
        
        # YOLOv8 output: [1, 84, 8400] → transpose to [8400, 84]
        output = output[0].T
        
        boxes = []
        scores = []
        for det in output:
            conf = det[4]
            if conf < 0.25:
                continue
            cx, cy, w, h = det[:4]
            x1 = (cx - w / 2) * imgsz
            y1 = (cy - h / 2) * imgsz
            x2 = (cx + w / 2) * imgsz
            y2 = (cy + h / 2) * imgsz
            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))
        
        predictions.append({"boxes": boxes, "scores": scores})
    
    gt = load_ground_truth(val_labels[:100], imgsz)
    metrics = compute_map(predictions, gt)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 2)
    metrics["fps"] = round(1000 / metrics["mean_latency_ms"], 1)
    metrics["model_size_mb"] = round(os.path.getsize(tflite_path) / (1024 * 1024), 2)
    
    return metrics

def load_ground_truth(label_paths, imgsz):
    gt = []
    for lbl_path in label_paths:
        boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, w, h = map(float, parts)
                        x1 = (cx - w / 2) * imgsz
                        y1 = (cy - h / 2) * imgsz
                        x2 = (cx + w / 2) * imgsz
                        y2 = (cy + h / 2) * imgsz
                        boxes.append([x1, y1, x2, y2])
        gt.append({"boxes": boxes})
    return gt

# ── Main ────────────────────────────────────────────────
def main():
    from tqdm import tqdm
    
    print("=" * 50)
    print("YOLOv8n Evaluation — Pascal VOC Person")
    print("=" * 50)
    
    # Find trained model
    pt_path = MODELS_DIR / "yolov8n_voc_person" / "weights" / "best.pt"
    if not pt_path.exists():
        print(f"Model not found at {pt_path}. Run 02_train.py first.")
        return
    
    # Load validation data
    val_img_dir = DATASET_DIR / "images" / "val"
    val_lbl_dir = DATASET_DIR / "labels" / "val"
    
    val_images = sorted(val_img_dir.glob("*.jpg"))
    val_labels = [val_lbl_dir / (p.stem + ".txt") for p in val_images]
    
    print(f"\nValidation: {len(val_images)} images")
    
    # E1: PyTorch FP32
    print("\n[1/3] Evaluating PyTorch FP32...")
    metrics_pytorch = evaluate_pytorch(pt_path, val_images, val_labels)
    print(f"  mAP@0.5: {metrics_pytorch['mAP@0.5']:.4f}")
    print(f"  Latency: {metrics_pytorch['mean_latency_ms']:.2f} ms")
    
    # Export to TFLite
    print("\n[Export] Converting to TFLite...")
    from ultralytics import YOLO
    model = YOLO(str(pt_path))
    
    tflite_dir = RESULTS_DIR / "tflite"
    tflite_dir.mkdir(exist_ok=True)
    
    # FP32 TFLite
    fp32_path = tflite_dir / "yolov8n_fp32.tflite"
    model.export(format="tflite", imgsz=640, half=False)
    # Move exported file
    exported = PROJECT_ROOT / "yolov8n_float32.tflite"
    if exported.exists():
        import shutil
        shutil.move(str(exported), str(fp32_path))
    
    # INT8 TFLite (PTQ)
    print("\n[Export] INT8 Quantization (PTQ)...")
    int8_path = tflite_dir / "yolov8n_int8.tflite"
    
    # Prepare calibration data
    calib_images = []
    for img_path in val_images[:200]:
        import cv2
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        calib_images.append(img.astype(np.float32) / 255.0)
    calib_images = np.array(calib_images)
    
    # Use tensorflow for INT8 conversion
    import tensorflow as tf
    
    # Convert FP32 TFLite to INT8
    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(PROJECT_ROOT / "models" / "yolov8n_voc_person" / "weights" / "best_saved_model")
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: [calib_images[i:i+1] for i in range(min(100, len(calib_images)))]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_int8 = converter.convert()
    with open(int8_path, "wb") as f:
        f.write(tflite_int8)
    print(f"  Saved: {int8_path} ({len(tflite_int8)} bytes)")
    
    # E2: TFLite FP32
    print("\n[2/3] Evaluating TFLite FP32...")
    metrics_fp32 = evaluate_tflite(fp32_path, val_images, val_labels)
    print(f"  mAP@0.5: {metrics_fp32['mAP@0.5']:.4f}")
    print(f"  Latency: {metrics_fp32['mean_latency_ms']:.2f} ms")
    
    # E3: TFLite INT8 (PTQ)
    print("\n[3/3] Evaluating TFLite INT8 (PTQ)...")
    metrics_int8 = evaluate_tflite(int8_path, val_images, val_labels)
    print(f"  mAP@0.5: {metrics_int8['mAP@0.5']:.4f}")
    print(f"  Latency: {metrics_int8['mean_latency_ms']:.2f} ms")
    
    # Save results
    results = {
        "experiment": "YOLOv8n Pascal VOC Person Detection",
        "dataset": "Pascal VOC 2012 (person class)",
        "date": datetime.now().isoformat(),
        "scenarios": {
            "E1_PyTorch_FP32": metrics_pytorch,
            "E2_TFLite_FP32": metrics_fp32,
            "E3_TFLite_INT8_PTQ": metrics_int8,
        },
        "comparison": {
            "mAP_drop_ptq_vs_fp32_tflite": round(
                metrics_fp32["mAP@0.5"] - metrics_int8["mAP@0.5"], 4
            ),
            "size_reduction_ptq": round(
                metrics_fp32["model_size_mb"] / max(metrics_int8["model_size_mb"], 0.01), 2
            ),
            "speedup_ptq": round(
                metrics_fp32["mean_latency_ms"] / max(metrics_int8["mean_latency_ms"], 0.01), 2
            ),
        }
    }
    
    results_path = RESULTS_DIR / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Results Summary")
    print(f"{'='*50}")
    for name, m in results["scenarios"].items():
        print(f"\n  {name}:")
        print(f"    mAP@0.5:  {m['mAP@0.5']:.4f}")
        print(f"    Latency:  {m['mean_latency_ms']:.2f} ms")
        print(f"    FPS:      {m['fps']:.1f}")
        print(f"    Size:     {m['model_size_mb']:.2f} MB")
    
    print(f"\n  PTQ vs FP32 (TFLite):")
    print(f"    mAP drop:   {results['comparison']['mAP_drop_ptq_vs_fp32_tflite']:.4f}")
    print(f"    Size ratio: {results['comparison']['size_reduction_ptq']}x")
    print(f"    Speedup:    {results['comparison']['speedup_ptq']}x")
    
    print(f"\nResults saved: {results_path}")
    return results

if __name__ == "__main__":
    main()
