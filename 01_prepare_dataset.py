#!/usr/bin/env python3
"""
COCO val2017 → YOLO Dataset (person class only)
~1GB download, 5K images, standard benchmark
"""

import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
from urllib.request import urlretrieve
import zipfile

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "_raw"

# COCO val2017
VAL_IMAGE_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
PERSON_CLASS_ID = 1
SEED = 42

random.seed(SEED)


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Annotations (~250MB)
    ann_zip = RAW_DIR / "annotations_trainval2017.zip"
    if not ann_zip.exists():
        print("[1/3] Downloading COCO annotations (~250MB)...")
        urlretrieve(ANN_URL, ann_zip)
    else:
        print("[1/3] Annotations exist.")

    ann_dir = RAW_DIR / "annotations"
    if not ann_dir.exists():
        with zipfile.ZipFile(ann_zip, "r") as z:
            z.extractall(RAW_DIR)

    # Val images (~1GB)
    img_zip = RAW_DIR / "val2017.zip"
    if not img_zip.exists():
        print("[2/3] Downloading COCO val2017 (~1GB)...")
        urlretrieve(VAL_IMAGE_URL, img_zip)
    else:
        print("[2/3] val2017 images exist.")

    img_dir = RAW_DIR / "val2017"
    if not img_dir.exists():
        with zipfile.ZipFile(img_zip, "r") as z:
            z.extractall(RAW_DIR)

    print("  Done downloading.")


def build_dataset():
    print("[3/3] Building YOLO dataset (person class)...")

    ann_file = RAW_DIR / "annotations" / "instances_val2017.json"
    img_dir = RAW_DIR / "val2017"

    with open(ann_file) as f:
        coco = json.load(f)

    # Index person annotations
    image_to_anns = defaultdict(list)
    for ann in tqdm(coco["annotations"], desc="  Indexing"):
        if ann["category_id"] == PERSON_CLASS_ID:
            bbox = ann["bbox"]
            if bbox[2] > 0 and bbox[3] > 0:
                image_to_anns[ann["image_id"]].append(ann)

    image_info = {img["id"]: img for img in coco["images"]}

    samples = []
    for img_id, anns in image_to_anns.items():
        img = image_info.get(img_id)
        if img is None:
            continue
        src = img_dir / img["file_name"]
        if not src.exists():
            continue

        w, h = img["width"], img["height"]
        labels = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        samples.append({"src": src, "name": img["file_name"], "labels": labels})

    print(f"  Person images: {len(samples)}")

    # Split (80/10/10)
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    for split_name, data in splits.items():
        img_out = DATASET_DIR / "images" / split_name
        lbl_out = DATASET_DIR / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for s in tqdm(data, desc=f"  {split_name}"):
            shutil.copy2(s["src"], img_out / s["name"])
            with open(lbl_out / (Path(s["name"]).stem + ".txt"), "w") as f:
                f.write("\n".join(s["labels"]))

        print(f"    {split_name}: {len(data)} images")

    # YAML
    yaml_content = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "person"},
    }
    yaml_path = PROJECT_ROOT / "coco_person.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"  YAML: {yaml_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("COCO val2017 → YOLO (person)")
    print("=" * 50)
    download()
    build_dataset()
