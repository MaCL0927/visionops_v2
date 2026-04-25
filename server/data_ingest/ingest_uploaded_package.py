#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ingest edge collector upload packages.

Usage:
  python server/data_ingest/ingest_uploaded_package.py
  python server/data_ingest/ingest_uploaded_package.py --package data/rk3588-001_CUST-001_20260425_144306.tar.gz
  python server/data_ingest/ingest_uploaded_package.py --incoming-dir data --keep-package

Input package example:
  data/rk3588-001_CUST-001_20260425_144306.tar.gz

Output:
  data/raw_collected/<batch_id>/
  data/collected_batches/index.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_DIRS = ["all_images", "positive", "negative"]


@dataclass
class BatchStatus:
    batch_id: str
    device_id: str
    customer_id: str
    captured_at: str
    source_package: str
    output_dir: str
    status: str
    all_images_count: int
    positive_count: int
    negative_count: int
    manifest_exists: bool
    collector_meta_exists: bool
    ingested_at: str
    message: str = ""


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_name(name: str) -> str:
    name = name.strip()
    name = name.replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", name)
    return name.strip("._") or "unknown"


def parse_batch_name(package_path: Path) -> tuple[str, str, str, str]:
    """
    Expected:
      rk3588-001_CUST-001_20260425_144306.tar.gz

    Return:
      batch_id, device_id, customer_id, captured_at
    """
    name = package_path.name
    if name.endswith(".tar.gz"):
        stem = name[:-7]
    else:
        stem = package_path.stem

    parts = stem.split("_")
    if len(parts) >= 4:
        device_id = safe_name(parts[0])
        customer_id = safe_name(parts[1])
        date_part = parts[2]
        time_part = parts[3]
        captured_at = f"{date_part}_{time_part}"
        batch_id = safe_name(stem)
        return batch_id, device_id, customer_id, captured_at

    batch_id = safe_name(stem)
    return batch_id, "unknown_device", "unknown_customer", "unknown_time"


def is_safe_tar_member(member: tarfile.TarInfo, dest_dir: Path) -> bool:
    """
    Prevent path traversal such as ../../evil.
    """
    member_path = dest_dir / member.name
    try:
        member_path.resolve().relative_to(dest_dir.resolve())
        return True
    except ValueError:
        return False


def safe_extract_tar_gz(package_path: Path, temp_dir: Path) -> None:
    if not tarfile.is_tarfile(package_path):
        raise ValueError(f"不是合法 tar 包: {package_path}")

    with tarfile.open(package_path, "r:gz") as tar:
        members = tar.getmembers()
        unsafe = [m.name for m in members if not is_safe_tar_member(m, temp_dir)]
        if unsafe:
            raise ValueError(f"压缩包包含不安全路径: {unsafe[:5]}")
        tar.extractall(temp_dir)


def find_dataset_root(extracted_dir: Path) -> Path:
    """
    The package may contain:
      all_images/
      positive/
      negative/

    Or:
      local_dataset/all_images/
      local_dataset/positive/
      local_dataset/negative/

    This function finds the directory containing required dataset dirs.
    """
    candidates = [extracted_dir]
    candidates.extend([p for p in extracted_dir.rglob("*") if p.is_dir()])

    for p in candidates:
        if all((p / d).is_dir() for d in REQUIRED_DIRS):
            return p

    raise FileNotFoundError(
        f"未找到数据集根目录，要求同时包含: {', '.join(REQUIRED_DIRS)}"
    )


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def copy_dataset_root(dataset_root: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"输出目录已存在: {output_dir}。如需覆盖，请加 --overwrite"
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for item in dataset_root.iterdir():
        dst = output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    for d in REQUIRED_DIRS:
        (output_dir / d).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def update_index(index_path: Path, status: BatchStatus) -> None:
    index = load_json(index_path, default={"batches": []})
    batches = index.get("batches", [])

    status_dict = asdict(status)
    existed = False
    for i, item in enumerate(batches):
        if item.get("batch_id") == status.batch_id:
            batches[i] = status_dict
            existed = True
            break

    if not existed:
        batches.append(status_dict)

    batches.sort(key=lambda x: x.get("ingested_at", ""), reverse=True)
    index["batches"] = batches
    index["updated_at"] = now_str()
    write_json(index_path, index)


def move_package(package_path: Path, incoming_dir: Path, status: str, keep_package: bool) -> Path | None:
    if keep_package:
        return None

    target_dir = incoming_dir / ("processed" if status == "success" else "failed")
    target_dir.mkdir(parents=True, exist_ok=True)

    dst = target_dir / package_path.name
    if dst.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = target_dir / f"{package_path.stem}_{timestamp}{package_path.suffix}"

    shutil.move(str(package_path), str(dst))
    return dst


def ingest_one_package(
    package_path: Path,
    incoming_dir: Path,
    raw_collected_dir: Path,
    index_path: Path,
    overwrite: bool,
    keep_package: bool,
) -> BatchStatus:
    package_path = package_path.resolve()
    if not package_path.exists():
        raise FileNotFoundError(f"压缩包不存在: {package_path}")

    batch_id, device_id, customer_id, captured_at = parse_batch_name(package_path)
    output_dir = raw_collected_dir / batch_id

    try:
        with tempfile.TemporaryDirectory(prefix="visionops_ingest_") as tmp:
            tmp_dir = Path(tmp)
            safe_extract_tar_gz(package_path, tmp_dir)
            dataset_root = find_dataset_root(tmp_dir)
            copy_dataset_root(dataset_root, output_dir, overwrite=overwrite)

        manifest_exists = (output_dir / "manifest.json").exists()
        collector_meta_exists = (output_dir / "collector_meta.jsonl").exists()

        status = BatchStatus(
            batch_id=batch_id,
            device_id=device_id,
            customer_id=customer_id,
            captured_at=captured_at,
            source_package=str(package_path),
            output_dir=str(output_dir),
            status="extracted",
            all_images_count=count_images(output_dir / "all_images"),
            positive_count=count_images(output_dir / "positive"),
            negative_count=count_images(output_dir / "negative"),
            manifest_exists=manifest_exists,
            collector_meta_exists=collector_meta_exists,
            ingested_at=now_str(),
            message="ok",
        )

        write_json(output_dir / "batch_status.json", asdict(status))
        update_index(index_path, status)
        move_package(package_path, incoming_dir, "success", keep_package=keep_package)

        return status

    except Exception as exc:
        failed_status = BatchStatus(
            batch_id=batch_id,
            device_id=device_id,
            customer_id=customer_id,
            captured_at=captured_at,
            source_package=str(package_path),
            output_dir=str(output_dir),
            status="failed",
            all_images_count=0,
            positive_count=0,
            negative_count=0,
            manifest_exists=False,
            collector_meta_exists=False,
            ingested_at=now_str(),
            message=str(exc),
        )
        update_index(index_path, failed_status)
        move_package(package_path, incoming_dir, "failed", keep_package=keep_package)
        raise


def discover_packages(incoming_dir: Path) -> list[Path]:
    packages = sorted(incoming_dir.glob("*.tar.gz"))
    return [
        p for p in packages
        if p.is_file()
        and "processed" not in p.parts
        and "failed" not in p.parts
    ]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest VisionOps edge collector packages.")
    parser.add_argument(
        "--package",
        type=str,
        default="",
        help="指定单个 tar.gz 包。默认扫描 incoming-dir 下所有 *.tar.gz。",
    )
    parser.add_argument(
        "--incoming-dir",
        type=str,
        default="data",
        help="上传包所在目录。当前测试包在 data/ 下，所以默认用 data。",
    )
    parser.add_argument(
        "--raw-collected-dir",
        type=str,
        default="data/raw_collected",
        help="解压后的批次数据目录。",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="data/collected_batches/index.json",
        help="批次索引文件路径。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果批次目录已存在，允许覆盖。",
    )
    parser.add_argument(
        "--keep-package",
        action="store_true",
        help="处理成功后保留原始压缩包，不移动到 processed/。",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()

    incoming_dir = Path(args.incoming_dir)
    raw_collected_dir = Path(args.raw_collected_dir)
    index_path = Path(args.index_path)

    incoming_dir.mkdir(parents=True, exist_ok=True)
    raw_collected_dir.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    (incoming_dir / "processed").mkdir(parents=True, exist_ok=True)
    (incoming_dir / "failed").mkdir(parents=True, exist_ok=True)

    if args.package:
        packages = [Path(args.package)]
    else:
        packages = discover_packages(incoming_dir)

    if not packages:
        print(f"[INFO] 未发现待处理压缩包: {incoming_dir}/*.tar.gz")
        return 0

    ok = 0
    failed = 0

    for package_path in packages:
        print(f"\n[INFO] 处理上传包: {package_path}")
        try:
            status = ingest_one_package(
                package_path=package_path,
                incoming_dir=incoming_dir,
                raw_collected_dir=raw_collected_dir,
                index_path=index_path,
                overwrite=args.overwrite,
                keep_package=args.keep_package,
            )
            ok += 1
            print("[OK] 解压完成")
            print(f"  batch_id:   {status.batch_id}")
            print(f"  device_id:  {status.device_id}")
            print(f"  customer:   {status.customer_id}")
            print(f"  output_dir: {status.output_dir}")
            print(f"  all_images: {status.all_images_count}")
            print(f"  positive:   {status.positive_count}")
            print(f"  negative:   {status.negative_count}")
        except Exception as exc:
            failed += 1
            print(f"[ERROR] 处理失败: {package_path}")
            print(f"  reason: {exc}")

    print("\n============================================================")
    print(f"完成: 成功 {ok} 个，失败 {failed} 个")
    print(f"索引: {index_path}")
    print("============================================================")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
