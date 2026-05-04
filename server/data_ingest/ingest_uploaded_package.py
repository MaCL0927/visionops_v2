#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ingest edge collector upload packages.

Usage:
  # 单包处理
  python server/data_ingest/ingest_uploaded_package.py \
    --package data/incoming/rk3588-001_CUST-001_20260425_144306.tar.gz

  # 多包合并处理：多个包会合并到一个 data/raw_collected/<merged_batch_id>/，并只同步一个 manifest
  python server/data_ingest/ingest_uploaded_package.py \
    --packages data/incoming/A.tar.gz data/incoming/B.tar.gz

Output:
  data/raw_collected/<batch_id>/
  data/collected_batches/index.json
  data/model_context/manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_DIRS = ["all_images", "positive", "negative"]
OPTIONAL_MERGE_DIRS = ["labels", "labels_auto"]


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
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", name)
    return name.strip("._") or "unknown"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


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
        captured_at = f"{parts[2]}_{parts[3]}"
        batch_id = safe_name(stem)
        return batch_id, device_id, customer_id, captured_at

    batch_id = safe_name(stem)
    return batch_id, "unknown_device", "unknown_customer", "unknown_time"


def is_safe_tar_member(member: tarfile.TarInfo, dest_dir: Path) -> bool:
    """Prevent path traversal such as ../../evil."""
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
    """
    candidates = [extracted_dir]
    candidates.extend([p for p in extracted_dir.rglob("*") if p.is_dir()])

    for p in candidates:
        if all((p / d).is_dir() for d in REQUIRED_DIRS):
            return p

    raise FileNotFoundError(f"未找到数据集根目录，要求同时包含: {', '.join(REQUIRED_DIRS)}")


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def copy_dataset_root(dataset_root: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在: {output_dir}。如需覆盖，请加 --overwrite")
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
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)


def copy_dir_contents(src_dir: Path, dst_dir: Path) -> int:
    """
    将 src_dir 下的文件复制到 dst_dir。
    当前项目约定不同包内不会有重名图片，因此这里不做重命名处理。
    """
    if not src_dir.exists():
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in src_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def append_text_file(src_file: Path, dst_file: Path) -> None:
    if not src_file.exists():
        return
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with src_file.open("r", encoding="utf-8", errors="ignore") as f_in, dst_file.open(
        "a", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            f_out.write(line.rstrip("\n") + "\n")


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
        stem = package_path.name[:-7] if package_path.name.endswith(".tar.gz") else package_path.stem
        dst = target_dir / f"{stem}_{timestamp}.tar.gz"

    shutil.move(str(package_path), str(dst))
    return dst


def sync_manifest_to_model_context(
    package_dir: Path,
    model_context_dir: Path = Path("data/model_context"),
) -> Path:
    """
    将当前采集包的 manifest.json 同步到固定位置：data/model_context/manifest.json。
    """
    src_manifest = package_dir / "manifest.json"
    if not src_manifest.exists():
        raise FileNotFoundError(f"未找到采集包 manifest.json: {src_manifest}")

    model_context_dir.mkdir(parents=True, exist_ok=True)
    dst_manifest = model_context_dir / "manifest.json"

    manifest = load_json(src_manifest, default={}) or {}
    package_name = package_dir.name

    device_id = (
        manifest.get("device_id")
        or manifest.get("equipment_id")
        or manifest.get("edge_device_id")
        or "unknown-device"
    )
    customer_id = (
        manifest.get("customer_id")
        or manifest.get("cust_id")
        or manifest.get("user_id")
        or "unknown-customer"
    )

    manifest["device_id"] = device_id
    manifest["customer_id"] = customer_id
    manifest["package_name"] = package_name
    manifest["source_manifest"] = str(src_manifest)
    manifest["model_context_updated_at"] = datetime.now().isoformat(timespec="seconds")

    write_json(dst_manifest, manifest)

    print(f"[OK] 已同步 manifest 到固定位置: {dst_manifest}")
    print(f"[INFO] device_id={device_id}, customer_id={customer_id}, package_name={package_name}")

    return dst_manifest


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
        sync_manifest_to_model_context(output_dir)
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


def auto_merged_batch_id(package_paths: list[Path]) -> str:
    _, device_id, customer_id, _ = parse_batch_name(package_paths[0])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return safe_name(f"{device_id}_{customer_id}_merged_{timestamp}")


def ingest_multiple_packages(
    package_paths: list[Path],
    incoming_dir: Path,
    raw_collected_dir: Path,
    index_path: Path,
    overwrite: bool,
    keep_package: bool,
    merged_batch_id: str = "",
) -> BatchStatus:
    """
    多包处理：把多个上传包合并到一个 raw_collected/<merged_batch_id>/，只生成一个 manifest.json。
    不做重名图片处理；如果不同包内存在同名文件，后复制的文件会覆盖前面的文件。
    """
    if len(package_paths) < 2:
        raise ValueError("多包处理至少需要 2 个压缩包")

    package_paths = [p.resolve() for p in package_paths]
    for p in package_paths:
        if not p.exists():
            raise FileNotFoundError(f"压缩包不存在: {p}")

    first_batch_id, first_device_id, first_customer_id, first_captured_at = parse_batch_name(package_paths[0])
    batch_id = safe_name(merged_batch_id) if merged_batch_id else auto_merged_batch_id(package_paths)
    output_dir = raw_collected_dir / batch_id

    try:
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(f"合并输出目录已存在: {output_dir}。如需覆盖，请加 --overwrite")
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        for d in REQUIRED_DIRS + OPTIONAL_MERGE_DIRS:
            (output_dir / d).mkdir(parents=True, exist_ok=True)

        sources: list[dict[str, Any]] = []

        for package_path in package_paths:
            sub_batch_id, device_id, customer_id, captured_at = parse_batch_name(package_path)
            print(f"[INFO] 合并上传包: {package_path}")

            with tempfile.TemporaryDirectory(prefix="visionops_merge_") as tmp:
                tmp_dir = Path(tmp)
                safe_extract_tar_gz(package_path, tmp_dir)
                dataset_root = find_dataset_root(tmp_dir)

                for d in REQUIRED_DIRS + OPTIONAL_MERGE_DIRS:
                    copy_dir_contents(dataset_root / d, output_dir / d)

                append_text_file(dataset_root / "collector_meta.jsonl", output_dir / "collector_meta.jsonl")

                src_manifest_path = dataset_root / "manifest.json"
                src_manifest = load_json(src_manifest_path, default={}) if src_manifest_path.exists() else {}
                sources.append(
                    {
                        "package_path": str(package_path),
                        "batch_id": sub_batch_id,
                        "device_id": device_id,
                        "customer_id": customer_id,
                        "captured_at": captured_at,
                        "manifest": src_manifest,
                    }
                )

        merged_manifest = {
            "schema_version": "visionops_collected_manifest_v1",
            "is_merged": True,
            "batch_id": batch_id,
            "package_name": batch_id,
            "device_id": first_device_id,
            "customer_id": first_customer_id,
            "captured_at": first_captured_at,
            "merged_at": datetime.now().isoformat(timespec="seconds"),
            "source_package_count": len(package_paths),
            "source_packages": [str(p) for p in package_paths],
            "sources": sources,
            "all_images_count": count_images(output_dir / "all_images"),
            "positive_count": count_images(output_dir / "positive"),
            "negative_count": count_images(output_dir / "negative"),
        }
        write_json(output_dir / "manifest.json", merged_manifest)

        status = BatchStatus(
            batch_id=batch_id,
            device_id=first_device_id,
            customer_id=first_customer_id,
            captured_at=first_captured_at,
            source_package=";".join(str(p) for p in package_paths),
            output_dir=str(output_dir),
            status="merged",
            all_images_count=count_images(output_dir / "all_images"),
            positive_count=count_images(output_dir / "positive"),
            negative_count=count_images(output_dir / "negative"),
            manifest_exists=True,
            collector_meta_exists=(output_dir / "collector_meta.jsonl").exists(),
            ingested_at=now_str(),
            message=f"merged {len(package_paths)} packages",
        )

        write_json(output_dir / "batch_status.json", asdict(status))
        update_index(index_path, status)
        sync_manifest_to_model_context(output_dir)

        for package_path in package_paths:
            move_package(package_path, incoming_dir, "success", keep_package=keep_package)

        return status

    except Exception as exc:
        failed_status = BatchStatus(
            batch_id=batch_id,
            device_id=first_device_id,
            customer_id=first_customer_id,
            captured_at=first_captured_at,
            source_package=";".join(str(p) for p in package_paths),
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
        for package_path in package_paths:
            if package_path.exists():
                move_package(package_path, incoming_dir, "failed", keep_package=keep_package)
        raise


def discover_packages(incoming_dir: Path) -> list[Path]:
    packages = sorted(incoming_dir.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime)
    return [p for p in packages if p.is_file() and "processed" not in p.parts and "failed" not in p.parts]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest VisionOps edge collector packages.")
    parser.add_argument(
        "--package",
        type=str,
        default="",
        help="指定单个 tar.gz 包。",
    )
    parser.add_argument(
        "--packages",
        nargs="+",
        default=[],
        help="指定一个或多个 tar.gz 包。传入多个包时会自动合并成一个 batch。",
    )
    parser.add_argument(
        "--merged-batch-id",
        type=str,
        default="",
        help="多包合并后的 batch_id。不传则自动生成。",
    )
    parser.add_argument(
        "--incoming-dir",
        type=str,
        default="data/incoming",
        help="上传包所在目录。默认 data/incoming。",
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

    if args.packages:
        packages = [Path(p) for p in args.packages]
    elif args.package:
        packages = [Path(args.package)]
    else:
        packages = discover_packages(incoming_dir)

    if not packages:
        print(f"[INFO] 未发现待处理压缩包: {incoming_dir}/*.tar.gz")
        return 0

    try:
        if len(packages) == 1:
            package_path = packages[0]
            print(f"\n[INFO] 单包处理: {package_path}")
            status = ingest_one_package(
                package_path=package_path,
                incoming_dir=incoming_dir,
                raw_collected_dir=raw_collected_dir,
                index_path=index_path,
                overwrite=args.overwrite,
                keep_package=args.keep_package,
            )
            print("[OK] 解压完成")
            print(f"  batch_id:   {status.batch_id}")
            print(f"  device_id:  {status.device_id}")
            print(f"  customer:   {status.customer_id}")
            print(f"  output_dir: {status.output_dir}")
            print(f"  all_images: {status.all_images_count}")
            print(f"  positive:   {status.positive_count}")
            print(f"  negative:   {status.negative_count}")
            print("  manifest:   data/model_context/manifest.json")
        else:
            print(f"\n[INFO] 多包合并处理: {len(packages)} 个压缩包")
            status = ingest_multiple_packages(
                package_paths=packages,
                incoming_dir=incoming_dir,
                raw_collected_dir=raw_collected_dir,
                index_path=index_path,
                overwrite=args.overwrite,
                keep_package=args.keep_package,
                merged_batch_id=args.merged_batch_id,
            )
            print("[OK] 合并解压完成")
            print(f"  merged_batch_id: {status.batch_id}")
            print(f"  output_dir:      {status.output_dir}")
            print(f"  packages:        {len(packages)}")
            print(f"  all_images:      {status.all_images_count}")
            print(f"  positive:        {status.positive_count}")
            print(f"  negative:        {status.negative_count}")
            print("  manifest:        data/model_context/manifest.json")

        print("\n============================================================")
        print("完成: 成功 1 个 batch，失败 0 个")
        print(f"索引: {index_path}")
        print("============================================================")
        return 0

    except Exception as exc:
        print("\n============================================================")
        print("完成: 成功 0 个 batch，失败 1 个")
        print(f"索引: {index_path}")
        print("============================================================")
        print(f"[ERROR] 处理失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
