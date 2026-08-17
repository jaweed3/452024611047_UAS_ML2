#!/bin/bash
# Download COCO val2017 + annotations
# Total: ~1.2GB, butuh ~30-60 menit tergantung koneksi

set -e

DATASET_DIR="$HOME/project/datasets/coco_val2017"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "=== COCO val2017 Downloader ==="
echo "Target: $DATASET_DIR"
echo ""

# 1. Download annotations (~250MB)
if [ -f "annotations/instances_val2017.json" ]; then
    echo "[1/2] Annotations sudah ada, skip."
else
    echo "[1/2] Downloading annotations (~250MB)..."
    curl -L -o annotations_trainval2017.zip \
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    echo "  Extracting..."
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
    echo "  Done."
fi

# 2. Download val images (~1GB)
if [ -d "val2017" ] && [ "$(ls -1 val2017/*.jpg 2>/dev/null | wc -l)" -eq 5000 ]; then
    echo "[2/2] val2017 images sudah ada, skip."
else
    echo "[2/2] Downloading val2017 images (~1GB)..."
    curl -L -o val2017.zip \
        "http://images.cocodataset.org/zips/val2017.zip"
    echo "  Extracting..."
    unzip -q val2017.zip
    rm val2017.zip
    echo "  Done."
fi

echo ""
echo "=== Selesai ==="
echo "Annotations: $DATASET_DIR/annotations/instances_val2017.json"
echo "Images: $DATASET_DIR/val2017/"
echo "Total images: $(ls -1 val2017/*.jpg | wc -l)"
echo ""
echo "Langkah selanjutnya: run run_experiments.py dengan path ke dataset ini."
