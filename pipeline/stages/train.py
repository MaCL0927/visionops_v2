"""
Stage 3: 模型训练
- 支持多种视觉任务（分类、检测、分割）
- 自动记录实验到MLflow
- 保存最优checkpoint
"""
import os
import json
import yaml
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model(architecture: str, num_classes: int, pretrained: bool):
    """根据配置获取模型（可扩展）"""
    arch = architecture.lower()

    if arch.startswith("yolov8"):
        # 使用ultralytics YOLOv8
        try:
            from ultralytics import YOLO
            size = arch.replace("yolov8", "") or "n"
            model = YOLO(f"yolov8{size}.pt" if pretrained else f"yolov8{size}.yaml")
            return model, "yolo"
        except ImportError:
            raise ImportError("请安装 ultralytics: pip install ultralytics")

    elif arch == "mobilenetv3":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, "torchvision"

    elif arch.startswith("efficientnet"):
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, "torchvision"

    else:
        raise ValueError(f"不支持的模型架构: {architecture}")


def train_torchvision(model, cfg: dict, device: str):
    """标准PyTorch训练循环"""
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    img_size = cfg["train"]["img_size"]
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.get("normalize", {}).get("mean", [0.485, 0.456, 0.406]),
            std=cfg.get("normalize", {}).get("std", [0.229, 0.224, 0.225])
        )
    ])

    train_dataset = datasets.ImageFolder("data/processed/train/", transform=transform)
    val_dataset = datasets.ImageFolder("data/processed/val/", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"],
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["train"]["batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    opt_name = cfg["train"].get("optimizer", "AdamW")
    if opt_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(),
                                       lr=cfg["train"]["lr"],
                                       weight_decay=cfg["train"].get("weight_decay", 0.0005))
    elif opt_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                     lr=cfg["train"]["lr"],
                                     momentum=0.9,
                                     weight_decay=cfg["train"].get("weight_decay", 0.0005))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )

    best_acc = 0.0
    patience_counter = 0
    patience = cfg["train"].get("early_stopping", {}).get("patience", 15)
    min_delta = cfg["train"].get("early_stopping", {}).get("min_delta", 0.001)

    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("models/metrics").mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler() if cfg.get("amp", False) and device == "cuda" else None

    all_metrics = []

    for epoch in range(cfg["train"]["epochs"]):
        # ── 训练阶段 ──
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)

        # ── 验证阶段 ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": optimizer.param_groups[0]["lr"]
        }
        all_metrics.append(epoch_metrics)

        # MLflow记录
        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
        }, step=epoch)

        print(f"Epoch [{epoch+1}/{cfg['train']['epochs']}] "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/checkpoints/best.pt")
            mlflow.log_artifact("models/checkpoints/best.pt")
            patience_counter = 0
            print(f"  ✓ 新最优模型已保存 (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1

        # 保存最新checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }, "models/checkpoints/last.pt")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 保存指标文件
    with open("models/metrics/train_metrics.json", "w") as f:
        json.dump({"best_val_acc": best_acc, "epochs_run": len(all_metrics),
                   "history": all_metrics}, f, indent=2)

    return best_acc


def main():
    # 加载配置
    data_cfg = load_config("pipeline/configs/data.yaml")
    train_cfg = load_config("pipeline/configs/train.yaml")
    cfg = {**data_cfg, **train_cfg}

    device = "cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu"
    print(f"使用设备: {device}")

    model, model_type = get_model(
        cfg["model"]["architecture"],
        cfg["model"]["num_classes"],
        cfg["model"].get("pretrained", True)
    )

    # 设置MLflow
    mlflow_cfg = cfg.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "visionops-training"))

    run_name = f"{cfg['model']['architecture']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        # 记录所有超参数
        mlflow.log_params({
            "architecture": cfg["model"]["architecture"],
            "num_classes": cfg["model"]["num_classes"],
            "epochs": cfg["train"]["epochs"],
            "batch_size": cfg["train"]["batch_size"],
            "lr": cfg["train"]["lr"],
            "optimizer": cfg["train"]["optimizer"],
            "img_size": str(cfg["train"]["img_size"]),
            "amp": cfg.get("amp", False),
            "device": device,
        })

        # 记录MLflow run ID供后续Stage使用
        with open("models/metrics/mlflow_run_id.txt", "w") as f:
            f.write(run.info.run_id)

        if model_type == "yolo":
            # YOLOv8训练
            results = model.train(
                data="pipeline/configs/yolo_dataset.yaml",
                epochs=cfg["train"]["epochs"],
                imgsz=cfg["train"]["img_size"][0],
                batch=cfg["train"]["batch_size"],
                lr0=cfg["train"]["lr"],
                project="models",
                name="yolo_run",
                exist_ok=True,
            )
            best_acc = float(results.results_dict.get("metrics/mAP50", 0))
            mlflow.log_metric("best_val_mAP50", best_acc)
        else:
            best_acc = train_torchvision(model, cfg, device)

        print(f"\n训练完成！最优验证精度: {best_acc:.4f}")
        print(f"MLflow Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
