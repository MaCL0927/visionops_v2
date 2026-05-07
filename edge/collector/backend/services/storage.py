#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import json
import re
import shutil
import subprocess
import tarfile
import shlex
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from backend.config import DATA_ROOT, DEVICE_ID, USER_ID, DEFAULT_DATASET_NAME, UPLOAD_TIMEOUT_SEC
from backend.services.settings_store import get_upload_runtime_config

SUBDIRS = ["all_images", "positive", "negative", "upload_packages"]
FOLDER_TO_SUBDIR = {"all": "all_images", "positive": "positive", "negative": "negative"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DATASET_NAME_PATTERN = re.compile(r"^[\w\-\u4e00-\u9fa5]{1,64}$")


def ensure_data_root() -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT


def sanitize_dataset_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("数据集名称不能为空")
    if not DATASET_NAME_PATTERN.match(name):
        raise ValueError("数据集名称只能包含中文、英文、数字、下划线和短横线，长度不超过 64")
    return name


def default_dataset_name() -> str:
    return sanitize_dataset_name(DEFAULT_DATASET_NAME)


def dataset_path(dataset: str = "") -> Path:
    name = sanitize_dataset_name(dataset or default_dataset_name())
    root = ensure_data_root()
    path = (root / name).resolve()
    if root not in path.parents and path != root:
        raise ValueError("非法数据集路径")
    return path


def ensure_dataset_dirs(dataset: str = "") -> Dict[str, str]:
    ds = dataset_path(dataset)
    for sub in SUBDIRS:
        (ds / sub).mkdir(parents=True, exist_ok=True)
    return {
        "data_root": str(DATA_ROOT),
        "dataset": ds.name,
        "dataset_root": str(ds),
        "all_images": str(ds / "all_images"),
        "positive": str(ds / "positive"),
        "negative": str(ds / "negative"),
        "upload_packages": str(ds / "upload_packages"),
    }


def ensure_default_dataset_dirs() -> Dict[str, str]:
    return ensure_dataset_dirs(default_dataset_name())


def list_datasets() -> List[Dict]:
    # 兼容旧接口：前端不再暴露数据集选择，只返回默认数据集。
    dirs = ensure_default_dataset_dirs()
    return [{"name": dirs["dataset"], "path": dirs["dataset_root"], "counts": get_counts(dirs["dataset"])}]


def create_dataset(dataset: str) -> Dict:
    # 兼容旧接口：仍可初始化默认目录，但不建议在工人界面使用。
    dirs = ensure_dataset_dirs(dataset or default_dataset_name())
    return {"name": dirs["dataset"], "dirs": dirs, "counts": get_counts(dirs["dataset"])}


def _safe_filename(filename: str) -> str:
    name = Path(filename or "").name
    if not name or name in {".", ".."}:
        raise ValueError("非法文件名")
    return name


def _safe_id(value: str, fallback: str) -> str:
    value = (value or fallback).strip()
    value = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fa5]+", "-", value)
    return value.strip("-") or fallback


def _folder_path(dataset: str, folder: str, create: bool = True) -> Path:
    if folder not in FOLDER_TO_SUBDIR:
        raise ValueError("folder 只能是 all / positive / negative")
    if create:
        ensure_dataset_dirs(dataset)
    return dataset_path(dataset) / FOLDER_TO_SUBDIR[folder]


def list_images(dataset: str, folder: str) -> List[Dict]:
    dirs = ensure_dataset_dirs(dataset or default_dataset_name())
    ds_name = dirs["dataset"]
    folder_path = _folder_path(ds_name, folder)
    items = []
    if not folder_path.exists():
        return items
    for p in sorted(folder_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stat = p.stat()
        items.append({
            "filename": p.name,
            "folder": folder,
            "dataset": ds_name,
            "size_bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "url": f"/api/datasets/{ds_name}/image/{folder}/{p.name}",
        })
    return items


def get_counts(dataset: str = "", create: bool = True) -> Dict[str, int]:
    ds_name = dataset or default_dataset_name()
    if create:
        ensure_dataset_dirs(ds_name)
    counts = {}
    for folder in FOLDER_TO_SUBDIR:
        folder_path = _folder_path(ds_name, folder, create=create)
        if not folder_path.exists():
            counts[folder] = 0
        else:
            counts[folder] = len([p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])
    return counts


def save_capture(dataset: str, image_data: str, folder: str = "all", device_id: str = "", user_id: str = "") -> Dict:
    dirs = ensure_dataset_dirs(dataset or default_dataset_name())
    if folder not in FOLDER_TO_SUBDIR:
        raise ValueError("folder 只能是 all / positive / negative")
    if not image_data or "," not in image_data:
        raise ValueError("未收到浏览器拍照图片，请允许摄像头权限或使用模拟画面")

    header, b64_data = image_data.split(",", 1)
    ext = ".jpg"
    if "image/png" in header:
        ext = ".png"
    elif "image/webp" in header:
        ext = ".webp"

    raw = base64.b64decode(b64_data)
    now = datetime.now()
    safe_device = _safe_id(device_id or DEVICE_ID, "device")
    safe_user = _safe_id(user_id or USER_ID, "user")
    filename = f"{safe_device}_{safe_user}_{now.strftime('%Y%m%d_%H%M%S_%f')}{ext}"

    # all_images 是完整采集底库，必须包含“取图 / 取正样本 / 取负样本”的全部图片。
    all_path = Path(dirs["all_images"]) / filename
    all_path.write_bytes(raw)
    out_path = all_path
    linked_all_image = None

    # 若直接采集为正/负样本，则同名再保存一份到对应标签目录，便于上传包按文件夹组织。
    if folder in {"positive", "negative"}:
        label_path = Path(dirs[FOLDER_TO_SUBDIR[folder]]) / filename
        label_path.write_bytes(raw)
        out_path = label_path
        linked_all_image = str(all_path)

    record = {
        "event": "capture",
        "dataset": dirs["dataset"],
        "folder": folder,
        "filename": filename,
        "device_id": safe_device,
        "user_id": safe_user,
        "created_at": now.isoformat(timespec="seconds"),
        "path": str(out_path),
        "all_image_path": str(all_path),
    }
    meta_file = Path(dirs["dataset_root"]) / "collector_meta.jsonl"
    with meta_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "filename": filename,
        "folder": folder,
        "dataset": dirs["dataset"],
        "device_id": safe_device,
        "user_id": safe_user,
        "url": f"/api/datasets/{dirs['dataset']}/image/{folder}/{filename}",
        "all_url": f"/api/datasets/{dirs['dataset']}/image/all/{filename}",
        "linked_all_image": linked_all_image,
        "size_bytes": out_path.stat().st_size,
    }



def save_capture_bytes(dataset: str, image_bytes: bytes, folder: str = "all", device_id: str = "", user_id: str = "", ext: str = ".jpg") -> Dict:
    """保存后端 latest_frame 的 JPEG 字节。用于 RTSP 单例读帧后的按钮采集。"""
    dirs = ensure_dataset_dirs(dataset or default_dataset_name())
    if folder not in FOLDER_TO_SUBDIR:
        raise ValueError("folder 只能是 all / positive / negative")
    if not image_bytes:
        raise ValueError("未收到摄像头图片数据")

    now = datetime.now()
    safe_device = _safe_id(device_id or DEVICE_ID, "device")
    safe_user = _safe_id(user_id or USER_ID, "user")
    ext = ext if ext.startswith(".") else f".{ext}"
    filename = f"{safe_device}_{safe_user}_{now.strftime('%Y%m%d_%H%M%S_%f')}{ext}"

    all_path = Path(dirs["all_images"]) / filename
    all_path.write_bytes(image_bytes)
    out_path = all_path
    linked_all_image = None

    if folder in {"positive", "negative"}:
        label_path = Path(dirs[FOLDER_TO_SUBDIR[folder]]) / filename
        label_path.write_bytes(image_bytes)
        out_path = label_path
        linked_all_image = str(all_path)

    record = {
        "event": "capture_backend_latest_frame",
        "dataset": dirs["dataset"],
        "folder": folder,
        "filename": filename,
        "device_id": safe_device,
        "user_id": safe_user,
        "created_at": now.isoformat(timespec="seconds"),
        "path": str(out_path),
        "all_image_path": str(all_path),
    }
    meta_file = Path(dirs["dataset_root"]) / "collector_meta.jsonl"
    with meta_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "filename": filename,
        "folder": folder,
        "dataset": dirs["dataset"],
        "device_id": safe_device,
        "user_id": safe_user,
        "url": f"/api/datasets/{dirs['dataset']}/image/{folder}/{filename}",
        "all_url": f"/api/datasets/{dirs['dataset']}/image/all/{filename}",
        "linked_all_image": linked_all_image,
        "size_bytes": out_path.stat().st_size,
    }



def save_labeled_capture(dataset: str, image_data: str, label: str, device_id: str = "", user_id: str = "") -> Dict:
    """分类采集确认后保存标签图片。

    与 save_capture(..., folder="positive/negative") 不同：
    这里只保存到 positive 或 negative 目录，不再同步保存到 all_images。
    用于“分类模式取图 -> 暂存 -> 选择合格/不合格”的流程。
    """
    if label not in {"positive", "negative"}:
        raise ValueError("label 只能是 positive 或 negative")
    if not image_data or "," not in image_data:
        raise ValueError("未收到暂存图片数据")

    header, b64_data = image_data.split(",", 1)
    ext = ".jpg"
    if "image/png" in header:
        ext = ".png"
    elif "image/webp" in header:
        ext = ".webp"
    raw = base64.b64decode(b64_data)
    return save_labeled_capture_bytes(dataset, raw, label, device_id, user_id, ext)


def save_labeled_capture_bytes(dataset: str, image_bytes: bytes, label: str, device_id: str = "", user_id: str = "", ext: str = ".jpg") -> Dict:
    """只保存分类标签图到 positive/negative，不写入 all_images。"""
    dirs = ensure_dataset_dirs(dataset or default_dataset_name())
    if label not in {"positive", "negative"}:
        raise ValueError("label 只能是 positive 或 negative")
    if not image_bytes:
        raise ValueError("未收到暂存图片数据")

    now = datetime.now()
    safe_device = _safe_id(device_id or DEVICE_ID, "device")
    safe_user = _safe_id(user_id or USER_ID, "user")
    ext = ext if ext.startswith(".") else f".{ext}"
    filename = f"{safe_device}_{safe_user}_{now.strftime('%Y%m%d_%H%M%S_%f')}{ext}"

    out_path = Path(dirs[FOLDER_TO_SUBDIR[label]]) / filename
    out_path.write_bytes(image_bytes)

    record = {
        "event": "classification_labeled_capture",
        "dataset": dirs["dataset"],
        "folder": label,
        "filename": filename,
        "device_id": safe_device,
        "user_id": safe_user,
        "created_at": now.isoformat(timespec="seconds"),
        "path": str(out_path),
        "all_image_path": None,
        "note": "saved_to_label_folder_only",
    }
    meta_file = Path(dirs["dataset_root"]) / "collector_meta.jsonl"
    with meta_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "filename": filename,
        "folder": label,
        "dataset": dirs["dataset"],
        "device_id": safe_device,
        "user_id": safe_user,
        "url": f"/api/datasets/{dirs['dataset']}/image/{label}/{filename}",
        "all_url": None,
        "linked_all_image": None,
        "size_bytes": out_path.stat().st_size,
    }

def label_image(dataset: str, filename: str, label: str, mode: str = "copy") -> Dict:
    # 兼容旧接口：仍支持从 all_images 复制到正/负样本。
    dirs = ensure_dataset_dirs(dataset or default_dataset_name())
    filename = _safe_filename(filename)
    if label not in {"positive", "negative"}:
        raise ValueError("label 只能是 positive 或 negative")
    if mode not in {"copy", "link"}:
        raise ValueError("mode 只能是 copy 或 link")
    ds = Path(dirs["dataset_root"])
    src = ds / "all_images" / filename
    if not src.exists():
        raise FileNotFoundError(f"全部图片目录中不存在：{filename}")
    dst = ds / label / filename
    opposite = ds / ("negative" if label == "positive" else "positive") / filename
    if opposite.exists():
        opposite.unlink()
    if dst.exists():
        dst.unlink()
    if mode == "link":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
            mode = "copy_fallback"
    else:
        shutil.copy2(src, dst)
    return {"filename": filename, "label": label, "mode": mode, "dataset": dirs["dataset"], "url": f"/api/datasets/{dirs['dataset']}/image/{label}/{filename}"}


def delete_image(dataset: str, folder: str, filename: str) -> Dict:
    """删除图片。

    v4.8 删除规则：
    - 在 all/全部图片 下删除：同步删除 all_images、positive、negative 中同名文件；
    - 在 positive/negative 下删除：也同步删除 all_images 中同名原图，并删除另一个标签目录中的同名文件。

    这样可以保证工人从任意预览入口删除一张图片时，采集底库和标签目录不会残留同名脏数据。
    """
    dirs = ensure_dataset_dirs(dataset or default_dataset_name())
    filename = _safe_filename(filename)
    if folder not in FOLDER_TO_SUBDIR:
        raise ValueError("folder 只能是 all / positive / negative")

    ds = Path(dirs["dataset_root"])
    # v4.8：无论从“全部图片 / 正样本 / 负样本”哪个入口删除，都删除三处同名文件。
    # 原因：取正/负样本时会同时保存到 all_images 和对应标签目录；如果只删标签目录，会导致 all_images 残留。
    targets = [
        ds / "all_images" / filename,
        ds / "positive" / filename,
        ds / "negative" / filename,
    ]

    deleted = []
    for p in targets:
        if p.exists() and p.is_file():
            p.unlink()
            deleted.append(str(p))
    if not deleted:
        raise FileNotFoundError(f"未找到待删除图片：{filename}")
    return {"dataset": dirs["dataset"], "folder": folder, "filename": filename, "deleted": deleted, "counts": get_counts(dirs["dataset"])}



def clear_capture_images(dataset: str = "") -> Dict:
    """一键清空当前采集目录中的图片。

    只清除 all_images / positive / negative 下的图片文件，不删除 upload_packages，
    也不删除 collector_meta.jsonl，避免影响历史上传包和采集日志。
    """
    dirs = ensure_dataset_dirs(dataset or default_dataset_name())
    deleted = {"all": 0, "positive": 0, "negative": 0}

    for folder, subdir in FOLDER_TO_SUBDIR.items():
        folder_path = Path(dirs[subdir])
        if not folder_path.exists():
            continue
        for p in folder_path.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                p.unlink()
                deleted[folder] += 1

    return {
        "dataset": dirs["dataset"],
        "deleted": deleted,
        "deleted_total": sum(deleted.values()),
        "counts": get_counts(dirs["dataset"]),
    }

def _run_cmd(cmd: List[str], timeout: int = UPLOAD_TIMEOUT_SEC) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def upload_package_to_pc(package_path: Path) -> Dict:
    """通过 SSH/SCP 将采集包上传到电脑端。

    设计目标参考原先 push.sh 的思路，但方向相反：
    - push.sh：电脑 -> 边缘板
    - 本函数：边缘板 -> 电脑

    为避免 Web 请求卡死，建议提前在 RK3588 上配置到电脑的 SSH 免密登录。
    """
    upload_cfg = get_upload_runtime_config()
    upload_enabled = bool(upload_cfg.get("enabled", True))
    upload_host = str(upload_cfg.get("host") or "").strip()
    upload_user = str(upload_cfg.get("user") or "pc").strip()
    upload_port = int(upload_cfg.get("port") or 22)
    upload_target_dir = str(upload_cfg.get("target_dir") or "").strip()
    upload_timeout_sec = int(upload_cfg.get("timeout_sec") or UPLOAD_TIMEOUT_SEC)

    if not upload_enabled:
        return {"enabled": False, "uploaded": False, "message": "远程上传已关闭，仅完成本地打包"}
    if not upload_host:
        return {
            "enabled": True,
            "uploaded": False,
            "message": "未配置服务端上传 IP，仅完成本地打包",
        }
    if not upload_target_dir:
        return {
            "enabled": True,
            "uploaded": False,
            "message": "未配置服务端接收目录，仅完成本地打包",
        }

    remote = f"{upload_user}@{upload_host}"
    ssh_base = [
        "ssh",
        "-p", str(upload_port),
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=8",
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    scp_base = [
        "scp",
        "-P", str(upload_port),
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=8",
        "-o", "StrictHostKeyChecking=accept-new",
    ]

    mkdir_cmd = ssh_base + [remote, f"mkdir -p {shlex.quote(upload_target_dir)}"]
    mkdir_res = _run_cmd(mkdir_cmd, timeout=min(30, upload_timeout_sec))
    if mkdir_res.returncode != 0:
        return {
            "enabled": True,
            "uploaded": False,
            "message": "远程目录创建失败，请检查电脑 SSH 服务、IP、用户名或免密登录",
            "target": f"{remote}:{upload_target_dir}",
            "stderr": mkdir_res.stderr.strip(),
        }

    remote_target = f"{remote}:{shlex.quote(upload_target_dir)}/"
    scp_cmd = scp_base + [str(package_path), remote_target]
    scp_res = _run_cmd(scp_cmd, timeout=upload_timeout_sec)
    if scp_res.returncode != 0:
        return {
            "enabled": True,
            "uploaded": False,
            "message": "上传电脑失败，请检查网络、SSH 免密、目标目录权限",
            "target": f"{remote}:{upload_target_dir}",
            "stderr": scp_res.stderr.strip(),
        }

    return {
        "enabled": True,
        "uploaded": True,
        "message": "已上传到电脑端数据目录",
        "target": f"{remote}:{upload_target_dir}/{package_path.name}",
    }


def create_upload_package(dataset: str, device_id: str, customer_id: str, contact_info: str = "", remark: str = "") -> Dict:
    dirs = ensure_dataset_dirs(dataset or default_dataset_name())
    if not device_id.strip():
        raise ValueError("设备 ID 必填")
    if not customer_id.strip():
        raise ValueError("客户 ID 必填")
    ds_name = dirs["dataset"]
    ds = Path(dirs["dataset_root"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_device = _safe_id(device_id, "device")
    safe_customer = _safe_id(customer_id, "customer")
    package_name = f"{safe_device}_{safe_customer}_{stamp}.tar.gz"
    package_path = ds / "upload_packages" / package_name
    manifest = {
        "dataset": ds_name,
        "device_id": device_id,
        "customer_id": customer_id,
        "contact_info": contact_info,
        "remark": remark,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "counts": get_counts(ds_name),
        "package_name": package_name,
    }
    manifest_path = ds / "manifest_latest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with tarfile.open(package_path, "w:gz") as tar:
        tar.add(ds / "all_images", arcname="all_images")
        tar.add(ds / "positive", arcname="positive")
        tar.add(ds / "negative", arcname="negative")
        meta_file = ds / "collector_meta.jsonl"
        if meta_file.exists():
            tar.add(meta_file, arcname="collector_meta.jsonl")
        tar.add(manifest_path, arcname="manifest.json")

    remote_upload = upload_package_to_pc(package_path)
    return {
        "dataset": ds_name,
        "package": package_name,
        "package_path": str(package_path),
        "counts": manifest["counts"],
        "remote_upload": remote_upload,
    }
