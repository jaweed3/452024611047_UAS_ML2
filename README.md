# YOLOv8n Optimization: PTQ vs Fine-tuning

**UAS Pembelajaran Mesin 2 — Universitas Darussalam Gontor**

**Nama:** Fatih Jawwad Al Mumtaz  
**NIM:** 452024611047  
**Email:** fatihalmumtaz76@student.cs.unida.gontor.ac.id

## Deskripsi

Eksperimen perbandingan dua strategi optimasi model deteksi objek YOLOv8n:

1. **E1: FP32 Baseline** — Model pretrained, fine-tuned pada COCO128
2. **E2: INT8 Quantization** — Post-training quantization via ONNX
3. **E3: Fine-tuning** — Frozen backbone (10 layer), training head saja

## Hasil

| Skenario | mAP@0.5 | Precision | Recall | Ukuran |
|----------|---------|-----------|--------|--------|
| E1: FP32 | 0.7257 | 0.7437 | 0.6534 | 6.25 MB |
| E2: INT8 | 0.7257 | 0.7437 | 0.6534 | 3.34 MB |
| E3: Fine-tune | 0.7588 | 0.8096 | 0.6859 | 6.23 MB |

**Temuan:**
- INT8 mengurangi ukuran 46.6%, tanpa drop akurasi
- Fine-tuning meningkatkan mAP +3.3%, presisi +6.6%

## Setup

### Prerequisites
- Python 3.12
- [uv](https://github.com/astral-sh/uv) (package manager)
- GPU dengan MPS support (Apple Silicon) atau CUDA

### Install Dependencies

```bash
uv sync
```

### Download Dataset (COCO128)

```bash
uv run -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('coco128.yaml')"
```

### Run Eksperimen

```bash
# Full pipeline (E1 + E2 + E3)
uv run run_experiments.py
```

Atau step-by-step:

```bash
# E1: Train FP32 baseline
uv run 02_train.py

# E2: Quantize to INT8
uv run 03_evaluate.py

# E3: Fine-tune dengan frozen backbone
# (otomatis dijalankan oleh run_experiments.py)
```

## Struktur Repository

```
├── 01_prepare_dataset.py    # Dataset preparation
├── 02_train.py              # Training pipeline
├── 03_evaluate.py           # Evaluation pipeline
├── run_experiments.py       # Full experiment pipeline
├── pyproject.toml           # Dependencies
├── paper/
│   ├── main.tex             # LaTeX source
│   └── 452024611047 - Fatih Jawwad Al Mumtaz.pdf
├── results/
│   ├── experiment_results.json
│   ├── e1_fp32/             # Training logs & curves
│   ├── e3_finetune/         # Training logs & curves
│   └── e2_int8.onnx         # Quantized model
└── README.md
```

## Paper

Paper IEEE Conference format ada di `paper/main.tex`. Compile dengan:

```bash
cd paper && tectonic main.tex
```

Atau upload ke [Overleaf](https://www.overleaf.com) dengan template IEEE Conference.

## License

Untuk keperluan akademik UAS Pembelajaran Mesin 2.
