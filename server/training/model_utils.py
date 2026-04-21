"""
模型构建工具函数
- 统一接口构建各类视觉模型
- 支持：YOLOv8 / MobileNetV3 / EfficientNet / ResNet
- ONNX导出辅助函数
"""
import torch
import torch.nn as nn
from typing import Tuple


SUPPORTED_ARCHITECTURES = [
    "yolov8n", "yolov8s", "yolov8m", "yolov8l",
    "mobilenetv3",
    "efficientnet_b0", "efficientnet_b1",
    "resnet18", "resnet34", "resnet50",
]


def build_model(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    统一模型构建入口
    返回 nn.Module（YOLOv8除外，请直接用 ultralytics.YOLO）
    """
    arch = architecture.lower()

    if arch.startswith("yolov8"):
        raise ValueError(
            "YOLOv8请直接使用 ultralytics.YOLO，不通过此函数构建。"
        )

    if arch == "mobilenetv3":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if arch == "efficientnet_b0":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if arch == "efficientnet_b1":
        from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = efficientnet_b1(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if arch == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "resnet34":
        from torchvision.models import resnet34, ResNet34_Weights
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(
        f"不支持的模型架构: {architecture}。"
        f"支持列表: {SUPPORTED_ARCHITECTURES}"
    )


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """返回 (total_params, trainable_params)"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model: nn.Module, input_size: list = None) -> dict:
    """简要模型信息"""
    total, trainable = count_parameters(model)
    info = {
        "total_params": total,
        "trainable_params": trainable,
        "param_size_mb": round(total * 4 / 1024 / 1024, 2),  # float32
    }
    if input_size:
        try:
            dummy = torch.randn(*input_size)
            with torch.no_grad():
                out = model(dummy)
            info["output_shape"] = list(out.shape)
        except Exception as e:
            info["output_shape_error"] = str(e)
    return info


def freeze_backbone(model: nn.Module, unfreeze_last_n: int = 1):
    """
    冻结主干，只训练最后N层（用于迁移学习微调）
    """
    params = list(model.parameters())
    for p in params[:-unfreeze_last_n]:
        p.requires_grad = False
    for p in params[-unfreeze_last_n:]:
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"冻结完成，可训练参数: {trainable:,}")
    return model
