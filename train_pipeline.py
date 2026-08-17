#!/usr/bin/env python3
"""UAS ML2 — YOLOv8n Training Pipeline (Hard Hat Detection)"""
import os, json, time, subprocess
from datetime import datetime
from pathlib import Path

PROJECT = Path(os.environ['USERPROFILE']) / 'project' / 'UAS_ML2'
RESULTS = PROJECT / 'results'
DATA_YAML = 'C:/dataset/data.yaml'
IMG_SIZE = 640
BATCH = 16
EPOCHS = 50
DEVICE = 0

results = {}

def train(name, freeze=0, lr=0.01, epochs=EPOCHS):
    print(f"\n{'='*50}")
    print(f"TRAINING: {name} (freeze={freeze}, lr={lr}, epochs={epochs})")
    print(f"{'='*50}")
    cmd = [
        'yolo', 'detect', 'train', 'model=yolov8n.pt',
        f'data={DATA_YAML}', f'epochs={epochs}', f'imgsz={IMG_SIZE}',
        f'batch={BATCH}', f'lr0={lr}', f'device={DEVICE}',
        f'project={RESULTS}', f'name={name}', 'exist_ok=True',
        'verbose=True', 'plots=True', 'save=True', 'patience=15',
    ]
    if freeze > 0:
        cmd.append(f'freeze={freeze}')
    t0 = time.time()
    subprocess.run(cmd, timeout=10800)
    print(f"  Done in {(time.time()-t0)/60:.1f} min")
    best = RESULTS / name / 'weights' / 'best.pt'
    return best if best.exists() else None

def evaluate(model_path, name):
    print(f"\n  Evaluating {name}...")
    cmd = ['yolo', 'detect', 'val', f'model={model_path}',
           f'data={DATA_YAML}', f'imgsz={IMG_SIZE}', f'device={DEVICE}',
           'plots=True', 'save_json=True']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    metrics = {}
    for line in r.stdout.split('\n'):
        parts = line.strip().split()
        if len(parts) >= 5 and 'all' in line.lower():
            try:
                metrics = {'precision': float(parts[1]), 'recall': float(parts[2]),
                           'mAP50': float(parts[3]), 'mAP50_95': float(parts[4])}
            except: pass
    metrics['size_mb'] = round(os.path.getsize(model_path) / (1024*1024), 2)
    print(f"  {name}: {metrics}")
    return metrics

# === MAIN ===
print(f"Start: {datetime.now()}")

# E1: FP32 50 epochs
e1 = train('E1_fp32_50ep', freeze=0, lr=0.01, epochs=50)
if e1:
    results['E1_fp32'] = evaluate(e1, 'E1_fp32')

# E2: INT8 PTQ
if e1:
    print("\n  INT8 quantization...")
    onnx_path = e1.parent.parent / f"{e1.stem}.onnx"
    if not onnx_path.exists():
        subprocess.run(['yolo', 'export', f'model={e1}', 'format=onnx',
                        f'imgsz={IMG_SIZE}', 'simplify=True'], timeout=300)
    if onnx_path.exists():
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            int8_path = RESULTS / 'E2_int8' / 'model_int8.onnx'
            os.makedirs(int8_path.parent, exist_ok=True)
            quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
            fp32_mb = os.path.getsize(onnx_path) / (1024*1024)
            int8_mb = os.path.getsize(int8_path) / (1024*1024)
            print(f"  FP32: {fp32_mb:.2f} MB -> INT8: {int8_mb:.2f} MB")
            import onnxruntime as ort, numpy as np
            sess = ort.InferenceSession(str(int8_path), providers=['CPUExecutionProvider'])
            inp = sess.get_inputs()[0]
            dummy = np.random.randn(*[1 if isinstance(d,str) else d for d in inp.shape]).astype(np.float32)
            for _ in range(10): sess.run(None, {inp.name: dummy})
            lats = []
            for _ in range(100):
                t0 = time.perf_counter()
                sess.run(None, {inp.name: dummy})
                lats.append((time.perf_counter()-t0)*1000)
            m = dict(results.get('E1_fp32', {}))
            m['mean_ms'] = round(float(np.mean(lats)), 2)
            m['fps'] = round(1000/float(np.mean(lats)), 1)
            m['size_mb'] = round(int8_mb, 2)
            results['E2_int8'] = m
            print(f"  INT8: {m['mean_ms']}ms, {m['fps']} FPS")
        except Exception as e:
            print(f"  INT8 error: {e}")

# E3: Fine-tune freeze10
e3 = train('E3_finetune_freeze10', freeze=10, lr=0.001, epochs=50)
if e3:
    results['E3_finetune_freeze10'] = evaluate(e3, 'E3_finetune_freeze10')

# E4: Fine-tune freeze5
e4 = train('E4_finetune_freeze5', freeze=5, lr=0.001, epochs=50)
if e4:
    results['E4_finetune_freeze5'] = evaluate(e4, 'E4_finetune_freeze5')

# Save results
output = {
    'experiment': 'YOLOv8n Optimization — Hard Hat Detection',
    'dataset': 'Hard Hat Detection (13,782 train, 2 classes: hardhat, no-hardhat)',
    'config': {'epochs': EPOCHS, 'batch': BATCH, 'imgsz': IMG_SIZE},
    'date': datetime.now().isoformat(),
    'results': results,
}
out_path = RESULTS / 'experiment_results.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*50}")
print("ALL DONE")
for k, v in results.items():
    print(f"  {k}: {v}")
print(f"Saved: {out_path}")
print(f"End: {datetime.now()}")
