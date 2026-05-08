#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps ROI Classification Pipeline Engine

第一版 roi_classification 运行时：
- stage1: detection RKNN 定位目标整体
- roi:    支持 full_box / relative_box / class_relative_box，可按检测类别使用精细 ROI
- stage2: classification RKNN 对 ROI 做分类
- FastAPI 接口保持 /health /infer /metrics /stats

典型启动：
python edge/inference/pipeline_engine.py \
  --pipeline-config /opt/visionops/models/<bundle>/pipeline.yaml \
  --host 0.0.0.0 \
  --port 8082 \
  --metrics-port 9091

pipeline.yaml 由 edge/deploy/push.sh --roi-classification 生成，典型结构：
pipeline_type: roi_classification
task: roi_classification
pipeline_name: rk3588-001_zt_roi_cls_20260506_163348

stage1:
  task: detection
  model_path: detector.rknn
  meta_path: detector.yaml
  input_size: [640, 640]
  num_classes: 1
  class_names: [tube]
  conf_threshold: 0.25
  nms_threshold: 0.45
  select_policy: conf_area
  target_class_id: 0
  target_class_name: tube

roi:
  # full_box: 检测框 + padding 后直接分类
  # class_relative_box: 先取检测框 + padding，再按检测类别应用相对裁剪框
  mode: class_relative_box
  coordinate: relative_to_padded_detection_box
  default:
    enabled: false
    mode: full_box
    padding_ratio: 0.0
    relative_box: {x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0}
  by_detector_class:
    "0:tube":
      enabled: true
      mode: relative_box
      padding_ratio: 0.0
      relative_box: {x1: 0.0, y1: 0.532364, x2: 1.0, y2: 0.793918}

stage2:
  task: classification
  model_path: classifier.rknn
  meta_path: classifier.yaml
  input_size: [224, 224]
  num_classes: 2
  class_names: [ng, ok]
  topk: 2
"""

import argparse
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    # 作为脚本运行时：/opt/visionops/edge/inference/pipeline_engine.py
    from engine import InferenceConfig, RKNNInferenceEngine, MetricsCollector
except Exception:
    # 作为模块运行时：python -m edge.inference.pipeline_engine
    from edge.inference.engine import InferenceConfig, RKNNInferenceEngine, MetricsCollector


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visionops.roi_classification")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("配置文件不存在: {}".format(path))
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML 顶层必须是 dict: {}".format(path))
    return data


def resolve_path(base_dir: Path, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def normalize_input_size(value: Any, default: List[int]) -> List[int]:
    if value is None:
        return default
    try:
        if isinstance(value, str):
            parts = value.replace(",", " ").split()
        elif isinstance(value, (list, tuple)):
            parts = list(value)
        else:
            return default
        if len(parts) != 2:
            return default
        h, w = int(parts[0]), int(parts[1])
        if h <= 0 or w <= 0:
            return default
        return [h, w]
    except Exception:
        return default


def normalize_class_names(value: Any, default_num: int) -> List[str]:
    if isinstance(value, list) and value:
        return [str(x) for x in value]
    if isinstance(value, dict) and value:
        def key_sort(k: Any) -> Any:
            try:
                return int(k)
            except Exception:
                return str(k)
        return [str(value[k]) for k in sorted(value.keys(), key=key_sort)]
    return [str(i) for i in range(max(1, int(default_num)))]


def load_stage_meta(stage_cfg: Dict[str, Any], bundle_dir: Path) -> Dict[str, Any]:
    meta_path = resolve_path(bundle_dir, stage_cfg.get("meta_path"))
    if not meta_path:
        return {}
    p = Path(meta_path)
    if not p.exists():
        logger.warning("meta 文件不存在，继续使用 pipeline.yaml 中的配置: %s", p)
        return {}
    try:
        return load_yaml(p)
    except Exception as exc:
        logger.warning("读取 meta 失败，继续使用 pipeline.yaml 中的配置: %s, err=%s", p, exc)
        return {}


def first_nonempty(*values: Any) -> Any:
    for v in values:
        if v is not None and str(v).strip() != "":
            return v
    return None


def build_stage_engine_config(
    stage_cfg: Dict[str, Any],
    meta_cfg: Dict[str, Any],
    bundle_dir: Path,
    default_task: str,
    default_input_size: List[int],
    npu_core: str,
    warmup_runs: int,
    metrics_port: int,
    debug_shapes: bool,
) -> InferenceConfig:
    model_path = resolve_path(bundle_dir, stage_cfg.get("model_path"))
    if not model_path:
        raise ValueError("{} 阶段缺少 model_path".format(default_task))

    task = str(first_nonempty(stage_cfg.get("task"), meta_cfg.get("task"), default_task))
    input_size = normalize_input_size(
        first_nonempty(stage_cfg.get("input_size"), meta_cfg.get("input_size")),
        default_input_size,
    )

    raw_names = first_nonempty(
        stage_cfg.get("class_names"),
        stage_cfg.get("names"),
        meta_cfg.get("class_names"),
        meta_cfg.get("names"),
    )
    raw_num = first_nonempty(stage_cfg.get("num_classes"), meta_cfg.get("num_classes"))
    try:
        num_classes = int(raw_num) if raw_num is not None else 0
    except Exception:
        num_classes = 0

    class_names = normalize_class_names(raw_names, max(num_classes, 1))
    if num_classes <= 0:
        num_classes = len(class_names)

    if len(class_names) != num_classes:
        logger.warning(
            "%s 类别数量不一致，按 class_names 长度修正: num_classes=%s, len(class_names)=%s",
            task, num_classes, len(class_names),
        )
        num_classes = len(class_names)

    conf_threshold = float(first_nonempty(stage_cfg.get("conf_threshold"), meta_cfg.get("conf_threshold"), 0.25))
    nms_threshold = float(first_nonempty(stage_cfg.get("nms_threshold"), meta_cfg.get("nms_threshold"), 0.45))
    topk = int(first_nonempty(stage_cfg.get("topk"), meta_cfg.get("topk"), min(5, num_classes)))

    return InferenceConfig(
        model_path=model_path,
        task=task,
        input_size=input_size,
        npu_core=npu_core,
        num_classes=num_classes,
        class_names=class_names,
        class_names_file=resolve_path(bundle_dir, stage_cfg.get("meta_path")),
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        topk=topk,
        metrics_port=metrics_port,
        warmup_runs=warmup_runs,
        debug_shapes=debug_shapes,
    )


def clip_xyxy(box: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(x) for x in box[:4]]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    return [x1, y1, x2, y2]


def crop_full_box_roi(
    image: np.ndarray,
    bbox: List[float],
    padding_ratio: float,
    min_width: int = 4,
    min_height: int = 4,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clip_xyxy(bbox, w, h)

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    pad_x = bw * float(padding_ratio)
    pad_y = bh * float(padding_ratio)

    rx1, ry1, rx2, ry2 = clip_xyxy(
        [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y],
        w,
        h,
    )

    ix1, iy1, ix2, iy2 = [int(round(v)) for v in [rx1, ry1, rx2, ry2]]
    ix1 = max(0, min(w - 1, ix1))
    iy1 = max(0, min(h - 1, iy1))
    ix2 = max(0, min(w, ix2))
    iy2 = max(0, min(h, iy2))

    if ix2 <= ix1 or iy2 <= iy1:
        return None, None
    if (ix2 - ix1) < int(min_width) or (iy2 - iy1) < int(min_height):
        return None, None

    roi = image[iy1:iy2, ix1:ix2].copy()
    return roi, [float(ix1), float(iy1), float(ix2), float(iy2)]




def normalize_relative_box(value: Any) -> Dict[str, float]:
    """
    将 relative_box 归一化到 [0,1]。

    relative_box 的坐标系是 base ROI：
      base ROI = detector bbox + padding
      final ROI = base ROI 内的相对裁剪框
    """
    if not isinstance(value, dict):
        value = {}

    def to_float(key: str, default: float) -> float:
        try:
            v = float(value.get(key, default))
        except Exception:
            v = float(default)
        return max(0.0, min(1.0, v))

    x1 = to_float("x1", 0.0)
    y1 = to_float("y1", 0.0)
    x2 = to_float("x2", 1.0)
    y2 = to_float("y2", 1.0)

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # 防止极小框导致空 ROI；这里保守地扩大到最小 2%。
    min_size = 0.02
    if (x2 - x1) < min_size:
        x2 = min(1.0, x1 + min_size)
        x1 = max(0.0, x2 - min_size)
    if (y2 - y1) < min_size:
        y2 = min(1.0, y1 + min_size)
        y1 = max(0.0, y2 - min_size)

    return {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
    }


def detector_class_key(pred: Dict[str, Any]) -> str:
    try:
        cid = int(pred.get("class_id"))
    except Exception:
        cid = -1
    cname = str(pred.get("class_name", cid)).strip() or str(cid)
    return "{}:{}".format(cid, cname)


def crop_image_by_xyxy(
    image: np.ndarray,
    bbox: List[float],
    min_width: int = 4,
    min_height: int = 4,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clip_xyxy(bbox, w, h)

    ix1, iy1, ix2, iy2 = [int(round(v)) for v in [x1, y1, x2, y2]]
    ix1 = max(0, min(w - 1, ix1))
    iy1 = max(0, min(h - 1, iy1))
    ix2 = max(0, min(w, ix2))
    iy2 = max(0, min(h, iy2))

    if ix2 <= ix1 or iy2 <= iy1:
        return None, None
    if (ix2 - ix1) < int(min_width) or (iy2 - iy1) < int(min_height):
        return None, None

    roi = image[iy1:iy2, ix1:ix2].copy()
    return roi, [float(ix1), float(iy1), float(ix2), float(iy2)]


def crop_roi_with_policy(
    image: np.ndarray,
    bbox: List[float],
    policy: Dict[str, Any],
    min_width: int = 4,
    min_height: int = 4,
) -> Tuple[Optional[np.ndarray], Optional[List[float]], Optional[List[float]], Dict[str, float]]:
    """
    根据 ROI policy 裁剪最终分类 ROI。

    返回：
      roi: 实际送入分类模型的图像
      final_bbox: 原图坐标系下最终分类 ROI
      base_bbox: 原图坐标系下 detector bbox + padding 后的 base ROI
      relative_box: base ROI 坐标系下的比例框
    """
    padding_ratio = float(policy.get("padding_ratio", 0.0))
    base_roi, base_bbox = crop_full_box_roi(
        image=image,
        bbox=bbox,
        padding_ratio=padding_ratio,
        min_width=min_width,
        min_height=min_height,
    )
    if base_roi is None or base_bbox is None:
        return None, None, None, normalize_relative_box(policy.get("relative_box"))

    mode = str(policy.get("mode") or "full_box").strip().lower()
    rel_box = normalize_relative_box(policy.get("relative_box"))

    if mode != "relative_box":
        return base_roi, base_bbox, base_bbox, rel_box

    bx1, by1, bx2, by2 = [float(x) for x in base_bbox]
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)

    final_bbox = [
        bx1 + bw * rel_box["x1"],
        by1 + bh * rel_box["y1"],
        bx1 + bw * rel_box["x2"],
        by1 + bh * rel_box["y2"],
    ]
    final_roi, final_bbox = crop_image_by_xyxy(
        image=image,
        bbox=final_bbox,
        min_width=min_width,
        min_height=min_height,
    )
    return final_roi, final_bbox, base_bbox, rel_box


def select_detection_prediction(
    predictions: List[Dict[str, Any]],
    target_class_id: Optional[int],
    target_class_name: Optional[str],
    select_policy: str,
    image_shape: Tuple[int, int, int],
) -> Optional[Dict[str, Any]]:
    if not predictions:
        return None

    filtered: List[Dict[str, Any]] = []
    target_name = str(target_class_name).strip() if target_class_name is not None else ""

    for pred in predictions:
        if target_name:
            if str(pred.get("class_name", "")) != target_name:
                continue
        elif target_class_id is not None:
            try:
                if int(pred.get("class_id")) != int(target_class_id):
                    continue
            except Exception:
                continue
        filtered.append(pred)

    if not filtered:
        return None

    h, w = image_shape[:2]
    frame_area = max(1.0, float(h * w))
    policy = str(select_policy or "conf_area").lower().strip()

    def score(pred: Dict[str, Any]) -> float:
        bbox = pred.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [float(x) for x in bbox[:4]]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        conf = float(pred.get("confidence", 0.0))
        if policy in {"highest_conf", "conf", "confidence"}:
            return conf
        if policy in {"largest_area", "area"}:
            return area
        return conf * 0.7 + min(area / frame_area, 1.0) * 0.3

    filtered.sort(key=score, reverse=True)
    return filtered[0]


class ROIPipelineConfig:
    def __init__(self, pipeline_config_path: str, npu_core: str, warmup_runs: int, metrics_port: int, debug_shapes: bool):
        self.pipeline_config_path = Path(pipeline_config_path).resolve()
        self.bundle_dir = self.pipeline_config_path.parent
        self.raw = load_yaml(self.pipeline_config_path)

        pipeline_type = str(self.raw.get("pipeline_type") or self.raw.get("task") or "").strip()
        if pipeline_type not in {"roi_classification", "ROI_CLASSIFICATION"}:
            raise ValueError("当前 pipeline_engine.py 第一版仅支持 pipeline_type/task=roi_classification，当前: {}".format(pipeline_type))

        self.pipeline_name = str(self.raw.get("pipeline_name") or self.pipeline_config_path.parent.name)
        self.stage1 = self.raw.get("stage1") or self.raw.get("detector") or {}
        self.stage2 = self.raw.get("stage2") or self.raw.get("classifier") or {}
        self.roi = self.raw.get("roi") or {}
        self.decision = self.raw.get("decision") or {}

        if not isinstance(self.stage1, dict) or not isinstance(self.stage2, dict):
            raise ValueError("pipeline.yaml 中 stage1/stage2 必须是 dict")

        det_meta = load_stage_meta(self.stage1, self.bundle_dir)
        cls_meta = load_stage_meta(self.stage2, self.bundle_dir)

        self.detector_config = build_stage_engine_config(
            stage_cfg=self.stage1,
            meta_cfg=det_meta,
            bundle_dir=self.bundle_dir,
            default_task="detection",
            default_input_size=[640, 640],
            npu_core=npu_core,
            warmup_runs=warmup_runs,
            metrics_port=metrics_port,
            debug_shapes=debug_shapes,
        )
        self.classifier_config = build_stage_engine_config(
            stage_cfg=self.stage2,
            meta_cfg=cls_meta,
            bundle_dir=self.bundle_dir,
            default_task="classification",
            default_input_size=[224, 224],
            npu_core=npu_core,
            warmup_runs=warmup_runs,
            metrics_port=metrics_port,
            debug_shapes=debug_shapes,
        )

        if self.detector_config.normalized_task() != "detection":
            raise ValueError("roi_classification stage1 当前仅支持 detection，当前: {}".format(self.detector_config.task))
        if self.classifier_config.normalized_task() != "classification":
            raise ValueError("roi_classification stage2 当前仅支持 classification，当前: {}".format(self.classifier_config.task))

        self.roi_mode = str(self.roi.get("mode") or "full_box").strip().lower()
        if self.roi_mode not in {"full_box", "relative_box", "class_relative_box"}:
            raise ValueError(
                "roi_classification 支持 roi.mode=full_box / relative_box / class_relative_box，当前: {}".format(
                    self.roi_mode
                )
            )

        default_roi = self.roi.get("default") if isinstance(self.roi.get("default"), dict) else {}
        self.padding_ratio = float(first_nonempty(default_roi.get("padding_ratio"), self.roi.get("padding_ratio"), 0.0))
        self.default_relative_box = normalize_relative_box(
            first_nonempty(default_roi.get("relative_box"), self.roi.get("relative_box"), {})
        )
        self.roi_by_detector_class = (
            self.roi.get("by_detector_class") if isinstance(self.roi.get("by_detector_class"), dict) else {}
        )

        self.min_roi_width = int(first_nonempty(self.roi.get("min_width"), default_roi.get("min_width"), 4))
        self.min_roi_height = int(first_nonempty(self.roi.get("min_height"), default_roi.get("min_height"), 4))
        self.select_policy = str(self.stage1.get("select_policy") or "conf_area")

        self.target_class_name = self.stage1.get("target_class_name")
        target_id = self.stage1.get("target_class_id")
        self.target_class_id: Optional[int]
        if self.target_class_name is not None and str(self.target_class_name).strip():
            # 类别名优先，与服务端 ROI 数据制作逻辑保持一致。
            self.target_class_id = None
        elif target_id is None or str(target_id).strip() == "":
            self.target_class_id = None
        else:
            self.target_class_id = int(target_id)

    def resolve_roi_policy(self, selected_detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据当前检测类别选择 ROI 裁剪策略。

        full_box:
          detection bbox + padding 后直接分类。
        relative_box:
          对所有检测类别使用同一个 relative_box。
        class_relative_box:
          按 "class_id:class_name" 查找 by_detector_class；
          如果没有命中，回退 default/full_box。
        """
        class_key = detector_class_key(selected_detection)

        if self.roi_mode == "relative_box":
            return {
                "pipeline_roi_mode": self.roi_mode,
                "mode": "relative_box",
                "enabled": True,
                "class_key": class_key,
                "matched_class_key": "",
                "padding_ratio": float(first_nonempty(self.roi.get("padding_ratio"), self.padding_ratio)),
                "relative_box": normalize_relative_box(self.roi.get("relative_box")),
                "source": "global_relative_box",
            }

        if self.roi_mode == "class_relative_box":
            class_id = selected_detection.get("class_id")
            class_name = str(selected_detection.get("class_name", "")).strip()
            candidate_keys = [class_key]
            if class_id is not None:
                candidate_keys.append(str(class_id))
            if class_name:
                candidate_keys.append(class_name)

            matched_key = ""
            matched_policy: Optional[Dict[str, Any]] = None
            for key in candidate_keys:
                entry = self.roi_by_detector_class.get(key)
                if isinstance(entry, dict):
                    matched_key = key
                    matched_policy = entry
                    break

            if matched_policy and bool(matched_policy.get("enabled", False)):
                return {
                    "pipeline_roi_mode": self.roi_mode,
                    "mode": str(matched_policy.get("mode") or "relative_box").strip().lower(),
                    "enabled": True,
                    "class_key": class_key,
                    "matched_class_key": matched_key,
                    "padding_ratio": float(first_nonempty(matched_policy.get("padding_ratio"), self.padding_ratio)),
                    "relative_box": normalize_relative_box(matched_policy.get("relative_box")),
                    "source": "by_detector_class",
                }

            default_roi = self.roi.get("default") if isinstance(self.roi.get("default"), dict) else {}
            default_enabled = bool(default_roi.get("enabled", False))
            default_mode = str(default_roi.get("mode") or ("relative_box" if default_enabled else "full_box")).strip().lower()
            if not default_enabled:
                default_mode = "full_box"
            return {
                "pipeline_roi_mode": self.roi_mode,
                "mode": default_mode,
                "enabled": default_enabled,
                "class_key": class_key,
                "matched_class_key": "",
                "padding_ratio": float(first_nonempty(default_roi.get("padding_ratio"), self.padding_ratio)),
                "relative_box": normalize_relative_box(default_roi.get("relative_box")),
                "source": "default",
            }

        return {
            "pipeline_roi_mode": self.roi_mode,
            "mode": "full_box",
            "enabled": False,
            "class_key": class_key,
            "matched_class_key": "",
            "padding_ratio": self.padding_ratio,
            "relative_box": self.default_relative_box,
            "source": "full_box",
        }


    def summary(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "pipeline_config": str(self.pipeline_config_path),
            "bundle_dir": str(self.bundle_dir),
            "task": "roi_classification",
            "stage1": {
                "task": self.detector_config.normalized_task(),
                "model_path": self.detector_config.model_path,
                "input_size": self.detector_config.input_size,
                "num_classes": self.detector_config.num_classes,
                "class_names": self.detector_config.class_names,
                "conf_threshold": self.detector_config.conf_threshold,
                "nms_threshold": self.detector_config.nms_threshold,
                "select_policy": self.select_policy,
                "target_class_id": self.target_class_id,
                "target_class_name": self.target_class_name,
            },
            "roi": {
                "mode": self.roi_mode,
                "padding_ratio": self.padding_ratio,
                "min_width": self.min_roi_width,
                "min_height": self.min_roi_height,
                "default_relative_box": self.default_relative_box,
                "by_detector_class_keys": sorted([str(k) for k in self.roi_by_detector_class.keys()]),
            },
            "stage2": {
                "task": self.classifier_config.normalized_task(),
                "model_path": self.classifier_config.model_path,
                "input_size": self.classifier_config.input_size,
                "num_classes": self.classifier_config.num_classes,
                "class_names": self.classifier_config.class_names,
                "topk": self.classifier_config.topk,
            },
        }


class ROICLassificationPipelineEngine:
    def __init__(self, cfg: ROIPipelineConfig):
        self.cfg = cfg
        self.detector = RKNNInferenceEngine(cfg.detector_config)
        self.classifier = RKNNInferenceEngine(cfg.classifier_config)
        self.metrics = MetricsCollector()
        self.is_loaded = False

    def load(self) -> bool:
        logger.info("加载 ROI Classification pipeline: %s", self.cfg.pipeline_name)
        logger.info("pipeline_config=%s", self.cfg.pipeline_config_path)

        det_ok = self.detector.load_model()
        if not det_ok:
            logger.error("检测模型加载失败: %s", self.cfg.detector_config.model_path)
            self.is_loaded = False
            return False

        cls_ok = self.classifier.load_model()
        if not cls_ok:
            logger.error("分类模型加载失败: %s", self.cfg.classifier_config.model_path)
            self.detector.release()
            self.is_loaded = False
            return False

        self.is_loaded = True
        logger.info("✓ ROI Classification pipeline 加载成功")
        return True

    def release(self) -> None:
        self.detector.release()
        self.classifier.release()
        self.is_loaded = False
        logger.info("ROI Classification pipeline 已释放")

    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("ROI Classification pipeline 未加载")

        t0 = time.perf_counter()
        try:
            det_t0 = time.perf_counter()
            det_result = self.detector.infer(image)
            det_latency = (time.perf_counter() - det_t0) * 1000.0

            det_predictions = det_result.get("predictions") or []
            selected = select_detection_prediction(
                predictions=det_predictions,
                target_class_id=self.cfg.target_class_id,
                target_class_name=self.cfg.target_class_name,
                select_policy=self.cfg.select_policy,
                image_shape=image.shape,
            )

            if selected is None:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self.metrics.record(latency_ms, success=True)
                return {
                    "task": "roi_classification",
                    "status": "no_target",
                    "final_decision": str(self.cfg.decision.get("no_target_policy", "NO_TARGET")),
                    "predictions": [],
                    "detector": {
                        "count": len(det_predictions),
                        "selected": None,
                        "predictions": det_predictions,
                        "latency_ms": round(det_latency, 2),
                    },
                    "classifier": None,
                    "roi": None,
                    "timing_ms": {
                        "detector": round(det_latency, 2),
                        "crop": 0.0,
                        "classifier": 0.0,
                        "total": round(latency_ms, 2),
                    },
                    "latency_ms": round(latency_ms, 2),
                }

            crop_t0 = time.perf_counter()
            roi_policy = self.cfg.resolve_roi_policy(selected)
            roi, roi_box, base_roi_box, relative_box = crop_roi_with_policy(
                image=image,
                bbox=selected.get("bbox") or [0, 0, 0, 0],
                policy=roi_policy,
                min_width=self.cfg.min_roi_width,
                min_height=self.cfg.min_roi_height,
            )
            crop_latency = (time.perf_counter() - crop_t0) * 1000.0

            if roi is None or roi_box is None:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self.metrics.record(latency_ms, success=True)
                return {
                    "task": "roi_classification",
                    "status": "bad_roi",
                    "final_decision": str(self.cfg.decision.get("bad_roi_policy", "REVIEW")),
                    "predictions": [],
                    "detector": {
                        "count": len(det_predictions),
                        "selected": selected,
                        "predictions": det_predictions,
                        "latency_ms": round(det_latency, 2),
                    },
                    "classifier": None,
                    "roi": {
                        "mode": roi_policy.get("mode", self.cfg.roi_mode),
                        "pipeline_mode": roi_policy.get("pipeline_roi_mode", self.cfg.roi_mode),
                        "padding_ratio": roi_policy.get("padding_ratio", self.cfg.padding_ratio),
                        "bbox": roi_box,
                        "base_bbox": base_roi_box,
                        "relative_box": relative_box,
                        "class_key": roi_policy.get("class_key", ""),
                        "matched_class_key": roi_policy.get("matched_class_key", ""),
                        "source": roi_policy.get("source", ""),
                    },
                    "timing_ms": {
                        "detector": round(det_latency, 2),
                        "crop": round(crop_latency, 2),
                        "classifier": 0.0,
                        "total": round(latency_ms, 2),
                    },
                    "latency_ms": round(latency_ms, 2),
                }

            cls_t0 = time.perf_counter()
            cls_result = self.classifier.infer(roi)
            cls_latency = (time.perf_counter() - cls_t0) * 1000.0

            prediction = cls_result.get("prediction")
            if not prediction:
                final_label = str(self.cfg.decision.get("low_cls_conf_policy", "REVIEW"))
                final_conf = 0.0
                final_class_id = -1
            else:
                final_label = str(prediction.get("class_name"))
                final_conf = float(prediction.get("confidence", 0.0))
                final_class_id = int(prediction.get("class_id", -1))

            x1, y1, x2, y2 = [float(x) for x in selected.get("bbox", [0, 0, 0, 0])[:4]]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            pipeline_prediction = {
                "class_id": final_class_id,
                "class_name": final_label,
                "confidence": round(final_conf, 6),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "center": [round(cx, 2), round(cy, 2)],
                "center_x": round(cx, 2),
                "center_y": round(cy, 2),
                "detector": selected,
                "classifier": prediction,
                "roi": {
                    "mode": roi_policy.get("mode", self.cfg.roi_mode),
                    "pipeline_mode": roi_policy.get("pipeline_roi_mode", self.cfg.roi_mode),
                    "bbox": [round(float(v), 2) for v in roi_box],
                    "base_bbox": [round(float(v), 2) for v in base_roi_box] if base_roi_box else None,
                    "shape": [int(x) for x in roi.shape],
                    "padding_ratio": roi_policy.get("padding_ratio", self.cfg.padding_ratio),
                    "relative_box": relative_box,
                    "class_key": roi_policy.get("class_key", ""),
                    "matched_class_key": roi_policy.get("matched_class_key", ""),
                    "source": roi_policy.get("source", ""),
                },
            }

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.metrics.record(latency_ms, success=True)

            return {
                "task": "roi_classification",
                "status": "ok",
                "final_decision": final_label,
                "final_label": final_label,
                "final_confidence": round(final_conf, 6),
                "predictions": [pipeline_prediction],
                "detector": {
                    "count": len(det_predictions),
                    "selected": selected,
                    "predictions": det_predictions,
                    "latency_ms": round(det_latency, 2),
                },
                "classifier": {
                    "prediction": prediction,
                    "topk": cls_result.get("topk", []),
                    "latency_ms": round(cls_latency, 2),
                },
                "roi": pipeline_prediction["roi"],
                "timing_ms": {
                    "detector": round(det_latency, 2),
                    "crop": round(crop_latency, 2),
                    "classifier": round(cls_latency, 2),
                    "total": round(latency_ms, 2),
                },
                "latency_ms": round(latency_ms, 2),
            }

        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.metrics.record(latency_ms, success=False)
            logger.exception("ROI Classification 推理异常")
            raise


def create_app(pipeline_config: ROIPipelineConfig):
    try:
        from fastapi import FastAPI, HTTPException, UploadFile, File
        from fastapi.responses import JSONResponse, PlainTextResponse
    except ImportError as exc:
        raise ImportError("请安装 fastapi 与 python-multipart") from exc

    pipeline = ROICLassificationPipelineEngine(pipeline_config)

    @asynccontextmanager
    async def lifespan(app: Any):
        logger.info("启动 ROI Classification 推理服务...")
        ok = pipeline.load()
        if not ok:
            logger.error("ROI Classification pipeline 加载失败")
        yield
        pipeline.release()
        logger.info("ROI Classification 推理服务已关闭")

    app = FastAPI(
        title="VisionOps ROI Classification Pipeline",
        description="RK3588 Detection + ROI Crop + Classification Inference Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        summary = pipeline_config.summary()
        return {
            "status": "ok" if pipeline.is_loaded else "error",
            "task": "roi_classification",
            "pipeline_name": pipeline_config.pipeline_name,
            "pipeline_config": str(pipeline_config.pipeline_config_path),
            "stage1": summary["stage1"],
            "roi": summary["roi"],
            "stage2": summary["stage2"],
            "detector_loaded": pipeline.detector.is_loaded,
            "classifier_loaded": pipeline.classifier.is_loaded,
            "detector_simulate_mode": getattr(pipeline.detector, "_simulate_mode", False),
            "classifier_simulate_mode": getattr(pipeline.classifier, "_simulate_mode", False),
        }

    @app.post("/infer")
    async def infer_endpoint(file: UploadFile = File(...)):
        import cv2

        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="无法解码图像")
        result = pipeline.infer(image)
        return JSONResponse(result)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return pipeline.metrics.prometheus_format()

    @app.get("/stats")
    async def stats():
        return {
            "pipeline": pipeline.metrics.get_stats(),
            "detector": pipeline.detector.metrics.get_stats(),
            "classifier": pipeline.classifier.metrics.get_stats(),
        }

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="VisionOps ROI Classification Pipeline Engine")
    parser.add_argument(
        "--pipeline-config",
        default=os.getenv("PIPELINE_CONFIG"),
        required=os.getenv("PIPELINE_CONFIG") is None,
        help="ROI classification pipeline.yaml 路径",
    )
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8082")))
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=int(os.getenv("METRICS_PORT", "9091")),
        help="保留参数，用于与 engine.py 启动参数一致；metrics endpoint 仍在当前 FastAPI 服务上提供",
    )
    parser.add_argument(
        "--npu-core",
        default=os.getenv("NPU_CORE", "auto"),
        choices=["auto", "core_0", "core_1", "core_2", "core_0_1", "core_0_1_2"],
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=int(os.getenv("WARMUP_RUNS", "3")),
    )
    parser.add_argument(
        "--debug-shapes",
        action="store_true",
        default=os.getenv("DEBUG_SHAPES", "1") == "1",
    )

    args = parser.parse_args()

    cfg = ROIPipelineConfig(
        pipeline_config_path=args.pipeline_config,
        npu_core=args.npu_core,
        warmup_runs=args.warmup_runs,
        metrics_port=args.metrics_port,
        debug_shapes=args.debug_shapes,
    )

    logger.info("启动配置: task=roi_classification, pipeline=%s, port=%s", args.pipeline_config, args.port)
    logger.info("pipeline summary: %s", cfg.summary())

    app = create_app(cfg)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
