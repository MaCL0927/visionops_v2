"""
Stage 3: 分类模型评估

输入：
models/checkpoints/best.pt
data/processed/val/

输出：
models/metrics/
├── eval_metrics.json
├── eval_report.json
└── confusion_matrix.png
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import torch
import torch.nn as nn
import mlflow
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.training.model_utils import build_model, count_parameters


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_torch_load(path: Path, map_location="cpu") -> dict:
    """
    兼容不同 PyTorch 版本的 checkpoint 加载。
    新版本 torch.load 可能会引入 weights_only 行为变化。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def safe_mlflow_log_artifact(local_path: str, artifact_path: str = None) -> None:
    """
    防止本地缺少 MinIO/S3 凭据时，MLflow artifact 上传失败导致评估中断。
    """
    try:
        if artifact_path is None:
            mlflow.log_artifact(local_path)
        else:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
    except Exception as e:
        print(f"[WARN] MLflow artifact 上传失败，已跳过: {local_path}")
        print(f"[WARN] 原因: {type(e).__name__}: {e}")


def build_eval_transform(img_size: List[int], mean: List[float], std: List[float]):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, List[int], List[int], List[List[float]]]:
    model.eval()

    total_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[List[float]] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += float(loss.item())
            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, all_targets, all_preds, all_probs


def measure_latency_ms(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup: int = 5,
    repeat: int = 30,
) -> float:
    """
    简单测量单张图片平均推理耗时。
    注意：这是 PyTorch 端评估耗时，不代表 RKNN 上板耗时。
    """
    model.eval()

    try:
        inputs, _ = next(iter(loader))
    except StopIteration:
        return 0.0

    inputs = inputs[:1].to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    return sum(times) / max(len(times), 1)


def save_confusion_matrix_png(
    cm,
    class_names: List[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    data_cfg = load_yaml("pipeline/configs/data.yaml")
    train_cfg_all = load_yaml("pipeline/configs/train.yaml")

    paths_cfg = data_cfg.get("paths", {})
    preprocess_cfg = data_cfg.get("preprocess", {})
    output_cfg = train_cfg_all.get("output", {})
    mlflow_cfg = train_cfg_all.get("mlflow", {})

    processed_dir = Path(paths_cfg.get("processed", "data/processed/"))
    val_dir = processed_dir / "val"

    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "models/checkpoints/"))
    metrics_dir = Path(output_cfg.get("metrics_dir", "models/metrics/"))

    checkpoint_path = checkpoint_dir / "best.pt"
    eval_metrics_path = metrics_dir / "eval_metrics.json"
    eval_report_path = metrics_dir / "eval_report.json"
    confusion_png_path = metrics_dir / "confusion_matrix.png"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"未找到最优模型: {checkpoint_path}\n"
            "请先运行：python pipeline/stages/train.py"
        )

    if not val_dir.exists():
        raise FileNotFoundError(
            f"未找到验证集目录: {val_dir}\n"
            "请先运行：python pipeline/stages/preprocess.py"
        )

    requested_device = str(train_cfg_all.get("device", "cuda")).lower()
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")

    ckpt = safe_torch_load(checkpoint_path, map_location="cpu")

    architecture = ckpt.get("architecture", train_cfg_all.get("model", {}).get("architecture", "mobilenetv3"))
    num_classes = int(ckpt.get("num_classes", train_cfg_all.get("model", {}).get("num_classes", 0)))
    class_names = ckpt.get("class_names", [])
    img_size = ckpt.get("img_size", train_cfg_all.get("train", {}).get("img_size", [224, 224]))
    normalize = ckpt.get(
        "normalize",
        preprocess_cfg.get(
            "normalize",
            {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        ),
    )

    mean = normalize.get("mean", [0.485, 0.456, 0.406])
    std = normalize.get("std", [0.229, 0.224, 0.225])

    if not class_names:
        class_names = sorted([p.name for p in val_dir.iterdir() if p.is_dir()])

    if num_classes <= 0:
        num_classes = len(class_names)

    if len(class_names) != num_classes:
        raise ValueError(
            f"类别数量不一致：num_classes={num_classes}, class_names={class_names}"
        )

    batch_size = int(train_cfg_all.get("train", {}).get("batch_size", 16))
    num_workers = int(preprocess_cfg.get("num_workers", 4))

    print("=" * 60)
    print("分类模型评估开始")
    print("=" * 60)
    print(f"checkpoint: {checkpoint_path}")
    print(f"模型结构: {architecture}")
    print(f"类别数: {num_classes}")
    print(f"类别名: {class_names}")
    print(f"验证目录: {val_dir}")
    print(f"输入尺寸: {img_size}")
    print(f"device: {device}")

    eval_transform = build_eval_transform(
        img_size=img_size,
        mean=mean,
        std=std,
    )

    val_dataset = datasets.ImageFolder(str(val_dir), transform=eval_transform)

    if list(val_dataset.classes) != class_names:
        raise ValueError(
            f"ImageFolder 类别顺序 {val_dataset.classes} 与 checkpoint 中 class_names {class_names} 不一致。\n"
            "请删除 data/processed 后重新运行 preprocess.py 和 train.py。"
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)

    criterion = nn.CrossEntropyLoss()

    eval_loss, y_true, y_pred, y_prob = evaluate_model(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    latency_ms = measure_latency_ms(
        model=model,
        loader=val_loader,
        device=device,
        warmup=5,
        repeat=30,
    )

    metrics_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix_png(
        cm=cm,
        class_names=class_names,
        output_path=confusion_png_path,
    )

    eval_metrics = {
        "task": "classification",
        "status": "success",
        "architecture": architecture,
        "num_classes": num_classes,
        "class_names": class_names,
        "checkpoint": str(checkpoint_path),
        "val_dir": str(val_dir),
        "num_val_samples": len(val_dataset),
        "eval_loss": round(float(eval_loss), 6),
        "accuracy": round(float(accuracy), 6),
        "precision_macro": round(float(precision_macro), 6),
        "recall_macro": round(float(recall_macro), 6),
        "f1_macro": round(float(f1_macro), 6),
        "precision_weighted": round(float(precision_weighted), 6),
        "recall_weighted": round(float(recall_weighted), 6),
        "f1_weighted": round(float(f1_weighted), 6),
        "latency_ms": round(float(latency_ms), 4),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "evaluated_at": datetime.now().isoformat(),
    }

    eval_report = {
        "task": "classification",
        "status": "success",
        "architecture": architecture,
        "num_classes": num_classes,
        "class_names": class_names,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": y_prob,
        "metrics": eval_metrics,
        "evaluated_at": datetime.now().isoformat(),
    }

    with open(eval_metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)

    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)

    # MLflow 记录：失败不影响本地评估结果
    try:
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "visionops-classification"))

        run_id_file = metrics_dir / "mlflow_run_id.txt"
        run_id = None
        if run_id_file.exists():
            run_id = run_id_file.read_text(encoding="utf-8").strip()

        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(
                    {
                        "eval_loss": float(eval_loss),
                        "accuracy": float(accuracy),
                        "precision_macro": float(precision_macro),
                        "recall_macro": float(recall_macro),
                        "f1_macro": float(f1_macro),
                        "latency_ms": float(latency_ms),
                    }
                )
                safe_mlflow_log_artifact(str(eval_metrics_path))
                safe_mlflow_log_artifact(str(eval_report_path))
                safe_mlflow_log_artifact(str(confusion_png_path))
        else:
            with mlflow.start_run(run_name=f"eval-{architecture}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
                mlflow.log_metrics(
                    {
                        "eval_loss": float(eval_loss),
                        "accuracy": float(accuracy),
                        "precision_macro": float(precision_macro),
                        "recall_macro": float(recall_macro),
                        "f1_macro": float(f1_macro),
                        "latency_ms": float(latency_ms),
                    }
                )
                safe_mlflow_log_artifact(str(eval_metrics_path))
                safe_mlflow_log_artifact(str(eval_report_path))
                safe_mlflow_log_artifact(str(confusion_png_path))

    except Exception as e:
        print(f"[WARN] MLflow 记录评估结果失败，已跳过。")
        print(f"[WARN] 原因: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("✓ 分类模型评估完成")
    print("=" * 60)
    print(f"验证样本数: {len(val_dataset)}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"precision_macro: {precision_macro:.4f}")
    print(f"recall_macro: {recall_macro:.4f}")
    print(f"f1_macro: {f1_macro:.4f}")
    print(f"latency_ms: {latency_ms:.4f}")
    print(f"评估指标: {eval_metrics_path}")
    print(f"评估报告: {eval_report_path}")
    print(f"混淆矩阵: {confusion_png_path}")


if __name__ == "__main__":
    main()
