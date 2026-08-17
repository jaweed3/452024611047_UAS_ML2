#!/usr/bin/env python3
"""
Generate paper figures from training logs:
1. Training curves comparison (loss, mAP)
2. Model size vs mAP trade-off
3. Precision-Recall comparison
"""

import json
from pathlib import Path
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
PAPER_DIR = PROJECT_ROOT / "paper"
PAPER_DIR.mkdir(exist_ok=True)


def load_csv(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def fig1_training_curves():
    e1_data = load_csv(RESULTS_DIR / "e1_fp32" / "results.csv")
    e3_data = load_csv(RESULTS_DIR / "e3_finetune" / "results.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Training Curves: E1 (FP32) vs E3 (Fine-tune)', fontsize=14, fontweight='bold')

    epochs_e1 = [int(r['epoch']) for r in e1_data]
    epochs_e3 = [int(r['epoch']) for r in e3_data]

    axes[0, 0].plot(epochs_e1, [float(r['train/box_loss']) for r in e1_data], 'b-o', label='E1: FP32', markersize=4)
    axes[0, 0].plot(epochs_e3, [float(r['train/box_loss']) for r in e3_data], 'r-s', label='E3: Fine-tune', markersize=4)
    axes[0, 0].set_title('Box Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs_e1, [float(r['train/cls_loss']) for r in e1_data], 'b-o', label='E1: FP32', markersize=4)
    axes[0, 1].plot(epochs_e3, [float(r['train/cls_loss']) for r in e3_data], 'r-s', label='E3: Fine-tune', markersize=4)
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs_e1, [float(r['metrics/mAP50(B)']) for r in e1_data], 'b-o', label='E1: FP32', markersize=4)
    axes[1, 0].plot(epochs_e3, [float(r['metrics/mAP50(B)']) for r in e3_data], 'r-s', label='E3: Fine-tune', markersize=4)
    axes[1, 0].set_title('mAP@0.5')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs_e1, [float(r['metrics/mAP50-95(B)']) for r in e1_data], 'b-o', label='E1: FP32', markersize=4)
    axes[1, 1].plot(epochs_e3, [float(r['metrics/mAP50-95(B)']) for r in e3_data], 'r-s', label='E3: Fine-tune', markersize=4)
    axes[1, 1].set_title('mAP@0.5:0.95')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('mAP')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PAPER_DIR / "fig1_training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def fig2_tradeoff():
    with open(RESULTS_DIR / "experiment_results.json") as f:
        results = json.load(f)

    scenarios = ['E1: FP32', 'E2: INT8', 'E3: Fine-tune']
    map50 = [
        results['scenarios']['E1_PyTorch_FP32']['map50'],
        results['scenarios']['E2_ONNX_INT8']['map50'],
        results['scenarios']['E3_Finetune_Freeze10']['map50'],
    ]
    sizes = [
        results['scenarios']['E1_PyTorch_FP32']['model_size_mb'],
        results['scenarios']['E2_ONNX_INT8']['model_size_mb'],
        results['scenarios']['E3_Finetune_Freeze10']['model_size_mb'],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Model Comparison: Accuracy vs Size', fontsize=14, fontweight='bold')

    colors = ['#2196F3', '#FF9800', '#4CAF50']

    bars1 = ax1.bar(scenarios, map50, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_title('mAP@0.5')
    ax1.set_ylabel('mAP')
    ax1.set_ylim(0.6, 0.8)
    for bar, val in zip(bars1, map50):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    bars2 = ax2.bar(scenarios, sizes, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_title('Model Size (MB)')
    ax2.set_ylabel('Size (MB)')
    ax2.set_ylim(0, 8)
    for bar, val in zip(bars2, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    out_path = PAPER_DIR / "fig2_tradeoff.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def fig3_precision_recall():
    with open(RESULTS_DIR / "experiment_results.json") as f:
        results = json.load(f)

    scenarios = ['E1: FP32', 'E2: INT8', 'E3: Fine-tune']
    precision = [
        results['scenarios']['E1_PyTorch_FP32']['precision'],
        results['scenarios']['E2_ONNX_INT8']['precision'],
        results['scenarios']['E3_Finetune_Freeze10']['precision'],
    ]
    recall = [
        results['scenarios']['E1_PyTorch_FP32']['recall'],
        results['scenarios']['E2_ONNX_INT8']['recall'],
        results['scenarios']['E3_Finetune_Freeze10']['recall'],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(scenarios))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], precision, width, label='Precision', color='#2196F3', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar([i + width/2 for i in x], recall, width, label='Recall', color='#FF9800', edgecolor='black', linewidth=0.5)

    ax.set_title('Precision and Recall Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xticks(list(x))
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_path = PAPER_DIR / "fig3_precision_recall.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_training_curves()
    fig2_tradeoff()
    fig3_precision_recall()
    print("Done!")
