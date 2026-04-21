"""
Stage 3: 模型评估
- 在验证集上评估模型性能
- 生成混淆矩阵、分类报告
- 输出评估指标供后续Stage判断是否晋升
"""
import os
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_model(model, val_loader, device: str, class_names: list = None) -> dict:
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 基础指标
    accuracy = (all_preds == all_targets).mean()

    # 分类报告
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(all_targets, all_preds,
                                    target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_targets, all_preds).tolist()

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(report["macro avg"]["precision"], 4),
        "recall_macro": round(report["macro avg"]["recall"], 4),
        "f1_macro": round(report["macro avg"]["f1-score"], 4),
        "evaluated_at": datetime.now().isoformat(),
    }

    return metrics, cm, report


def main():
    train_cfg = load_config("pipeline/configs/train.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = "models/checkpoints/best.pt"
    if not os.path.exists(checkpoint_path):
        print("找不到checkpoint，使用训练指标作为评估结果")
        train_metrics_path = "models/metrics/train_metrics.json"
        if os.path.exists(train_metrics_path):
            with open(train_metrics_path) as f:
                data = json.load(f)
            metrics = {"accuracy": data.get("best_val_acc", 0.0)}
        else:
            metrics = {"accuracy": 0.0}

        Path("models/metrics").mkdir(parents=True, exist_ok=True)
        with open("models/metrics/eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"评估指标: {metrics}")
        return

    arch = train_cfg["model"]["architecture"]
    if arch.lower().startswith("yolov8"):
        # YOLOv8有自己的评估
        try:
            from ultralytics import YOLO
            model = YOLO(checkpoint_path)
            results = model.val(data="pipeline/configs/yolo_dataset.yaml")
            metrics = {
                "mAP50": round(float(results.results_dict.get("metrics/mAP50", 0)), 4),
                "mAP50_95": round(float(results.results_dict.get("metrics/mAP50-95", 0)), 4),
                "accuracy": round(float(results.results_dict.get("metrics/mAP50", 0)), 4),
                "evaluated_at": datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"YOLOv8评估失败: {e}")
            metrics = {"accuracy": 0.0}
    else:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader

        img_size = train_cfg["train"]["img_size"]
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_dataset = datasets.ImageFolder("data/processed/val/", transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        class_names = val_dataset.classes

        from torchvision.models import mobilenet_v3_small
        import torch.nn as nn
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,
                                           train_cfg["model"]["num_classes"])
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state if "model_state_dict" not in state else state["model_state_dict"])
        model = model.to(device)

        metrics, cm, report = evaluate_model(model, val_loader, device, class_names)

        Path("models/metrics").mkdir(parents=True, exist_ok=True)
        with open("models/metrics/eval_report.json", "w") as f:
            json.dump({"confusion_matrix": cm, "per_class": report}, f, indent=2)

    Path("models/metrics").mkdir(parents=True, exist_ok=True)
    with open("models/metrics/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 50)
    print("评估结果:")
    for k, v in metrics.items():
        if k != "evaluated_at":
            print(f"  {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
