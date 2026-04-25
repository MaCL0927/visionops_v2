"""
Stage 2: 分类模型训练

输入：
data/processed/
├── train/
│   ├── class_a/
│   └── class_b/
└── val/
    ├── class_a/
    └── class_b/

输出：
models/checkpoints/
├── best.pt
└── last.pt

models/metrics/
├── train_metrics.json
└── mlflow_run_id.txt
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 允许从仓库根目录导入 server.training.model_utils
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.training.model_utils import build_model, count_parameters, model_summary


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_names(processed_dir: Path) -> List[str]:
    """
    优先读取 preprocess.py 生成的 class_names.yaml。
    如果不存在，则回退到 ImageFolder 的目录顺序。
    """
    class_names_file = processed_dir / "class_names.yaml"

    if class_names_file.exists():
        with open(class_names_file, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        class_names = payload.get("class_names", [])
        if class_names:
            return list(class_names)

    train_dir = processed_dir / "train"
    if train_dir.exists():
        return sorted([p.name for p in train_dir.iterdir() if p.is_dir()])

    return []


def build_transforms(img_size: List[int], mean: List[float], std: List[float]):
    """
    分类训练 transform。
    注意：preprocess.py 已经 resize 时，这里 Resize 不会有副作用。
    """
    return {
        "train": transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        "val": transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    architecture: str,
    num_classes: int,
    class_names: List[str],
    img_size: List[int],
    normalize: Dict[str, List[float]],
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    best_val_acc: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "architecture": architecture,
            "num_classes": num_classes,
            "class_names": class_names,
            "img_size": img_size,
            "normalize": normalize,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "saved_at": datetime.now().isoformat(),
        },
        path,
    )


def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += float(loss.item())
            preds = outputs.argmax(dim=1)
            total_correct += int((preds == targets).sum().item())
            total_count += int(targets.size(0))

    avg_loss = total_loss / max(len(loader), 1)
    acc = total_correct / max(total_count, 1)

    return avg_loss, acc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
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

        total_loss += float(loss.item())
        preds = outputs.argmax(dim=1)
        total_correct += int((preds == targets).sum().item())
        total_count += int(targets.size(0))

    avg_loss = total_loss / max(len(loader), 1)
    acc = total_correct / max(total_count, 1)

    return avg_loss, acc


def build_optimizer(model: nn.Module, train_cfg: dict):
    opt_name = str(train_cfg.get("optimizer", "AdamW")).lower()
    lr = float(train_cfg.get("lr", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0005))

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError(f"不支持的优化器: {train_cfg.get('optimizer')}")


def build_scheduler(optimizer, train_cfg: dict):
    scheduler_name = str(train_cfg.get("lr_scheduler", "cosine")).lower()
    epochs = int(train_cfg.get("epochs", 100))

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if scheduler_name in {"none", "null", ""}:
        return None

    raise ValueError(f"不支持的 lr_scheduler: {train_cfg.get('lr_scheduler')}")


def main() -> None:
    data_cfg = load_config("pipeline/configs/data.yaml")
    train_cfg_all = load_config("pipeline/configs/train.yaml")

    paths_cfg = data_cfg.get("paths", {})
    preprocess_cfg = data_cfg.get("preprocess", {})
    model_cfg = train_cfg_all.get("model", {})
    train_cfg = train_cfg_all.get("train", {})
    early_cfg = train_cfg_all.get("early_stopping", {})
    mlflow_cfg = train_cfg_all.get("mlflow", {})
    output_cfg = train_cfg_all.get("output", {})

    processed_dir = Path(paths_cfg.get("processed", "data/processed/"))
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "未找到 data/processed/train 或 data/processed/val。\n"
            "请先运行：python pipeline/stages/preprocess.py"
        )

    architecture = str(model_cfg.get("architecture", "mobilenetv3"))
    configured_num_classes = int(model_cfg.get("num_classes", 0))
    pretrained = bool(model_cfg.get("pretrained", True))

    img_size = train_cfg.get("img_size", preprocess_cfg.get("img_size", [224, 224]))
    normalize = preprocess_cfg.get(
        "normalize",
        {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    )
    mean = normalize.get("mean", [0.485, 0.456, 0.406])
    std = normalize.get("std", [0.229, 0.224, 0.225])

    class_names = load_class_names(processed_dir)
    if not class_names:
        raise RuntimeError("未识别到类别名，请检查 data/processed/train/ 下是否有类别子目录。")

    detected_num_classes = len(class_names)
    if configured_num_classes and configured_num_classes != detected_num_classes:
        raise ValueError(
            f"train.yaml 中 num_classes={configured_num_classes}，"
            f"但数据集中识别到 {detected_num_classes} 类：{class_names}。\n"
            "请修改 pipeline/configs/train.yaml 中的 model.num_classes。"
        )

    num_classes = detected_num_classes

    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 100))
    num_workers = int(preprocess_cfg.get("num_workers", 4))

    requested_device = str(train_cfg_all.get("device", "cuda")).lower()
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "models/checkpoints/"))
    metrics_dir = Path(output_cfg.get("metrics_dir", "models/metrics/"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("分类模型训练开始")
    print("=" * 60)
    print(f"模型结构: {architecture}")
    print(f"类别数: {num_classes}")
    print(f"类别名: {class_names}")
    print(f"训练目录: {train_dir}")
    print(f"验证目录: {val_dir}")
    print(f"输入尺寸: {img_size}")
    print(f"batch_size: {batch_size}")
    print(f"epochs: {epochs}")
    print(f"device: {device}")

    tfms = build_transforms(img_size=img_size, mean=mean, std=std)

    train_dataset = datasets.ImageFolder(str(train_dir), transform=tfms["train"])
    val_dataset = datasets.ImageFolder(str(val_dir), transform=tfms["val"])

    # ImageFolder 会按类别名排序，这里检查和 class_names.yaml 保持一致
    if list(train_dataset.classes) != class_names:
        raise ValueError(
            f"ImageFolder 类别顺序 {train_dataset.classes} 与 class_names.yaml {class_names} 不一致。\n"
            "请删除 data/processed 后重新运行 preprocess.py。"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
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
        pretrained=pretrained,
    )
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"参数量: total={total_params:,}, trainable={trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    amp_enabled = bool(train_cfg_all.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None

    patience = int(early_cfg.get("patience", 15))
    min_delta = float(early_cfg.get("min_delta", 0.001))
    early_enabled = bool(early_cfg.get("enabled", True))

    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "visionops-classification"))

    run_name = f"{architecture}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    with mlflow.start_run(run_name=run_name) as run:
        run_id_file = metrics_dir / "mlflow_run_id.txt"
        with open(run_id_file, "w", encoding="utf-8") as f:
            f.write(run.info.run_id)

        mlflow.log_params(
            {
                "task": "classification",
                "architecture": architecture,
                "num_classes": num_classes,
                "class_names": ",".join(class_names),
                "pretrained": pretrained,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": train_cfg.get("lr"),
                "optimizer": train_cfg.get("optimizer"),
                "lr_scheduler": train_cfg.get("lr_scheduler"),
                "img_size": str(img_size),
                "amp": amp_enabled,
                "device": str(device),
                "total_params": total_params,
                "trainable_params": trainable_params,
            }
        )

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
            )

            val_loss, val_acc = evaluate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            if scheduler is not None:
                scheduler.step()

            lr = optimizer.param_groups[0]["lr"]

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "lr": lr,
            }
            history.append(epoch_metrics)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": lr,
                },
                step=epoch,
            )

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, lr={lr:.6g}"
            )

            improved = val_acc > best_val_acc + min_delta
            if improved:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0

                save_checkpoint(
                    path=checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    architecture=architecture,
                    num_classes=num_classes,
                    class_names=class_names,
                    img_size=img_size,
                    normalize=normalize,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    best_val_acc=best_val_acc,
                )

                mlflow.log_artifact(str(checkpoint_dir / "best.pt"))
                print(f"  ✓ 新最优模型已保存: val_acc={best_val_acc:.4f}")
            else:
                patience_counter += 1

            save_checkpoint(
                path=checkpoint_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                architecture=architecture,
                num_classes=num_classes,
                class_names=class_names,
                img_size=img_size,
                normalize=normalize,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                best_val_acc=best_val_acc,
            )

            if early_enabled and patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        summary = {
            "task": "classification",
            "status": "success",
            "architecture": architecture,
            "num_classes": num_classes,
            "class_names": class_names,
            "img_size": img_size,
            "normalize": normalize,
            "epochs_configured": epochs,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_val_acc": round(best_val_acc, 6),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "mlflow_run_id": run.info.run_id,
            "checkpoint_best": str(checkpoint_dir / "best.pt"),
            "checkpoint_last": str(checkpoint_dir / "last.pt"),
            "history": history,
            "finished_at": datetime.now().isoformat(),
        }

        with open(metrics_dir / "train_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_artifact(str(metrics_dir / "train_metrics.json"))

    print("\n" + "=" * 60)
    print("✓ 分类模型训练完成")
    print("=" * 60)
    print(f"最优 epoch: {best_epoch}")
    print(f"最优验证准确率: {best_val_acc:.4f}")
    print(f"best checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"last checkpoint: {checkpoint_dir / 'last.pt'}")
    print(f"训练指标: {metrics_dir / 'train_metrics.json'}")


if __name__ == "__main__":
    main()
