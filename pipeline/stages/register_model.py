"""
Stage 6: 模型注册到MLflow Model Registry
- 读取评估指标，判断是否满足晋升阈值
- 注册到MLflow Registry并标记版本
- 准备部署元数据
"""
import os
import sys
import json
import yaml
import mlflow
import mlflow.pyfunc
from pathlib import Path
from datetime import datetime
from mlflow.tracking import MlflowClient


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_eval_metrics() -> dict:
    metrics_path = "models/metrics/eval_metrics.json"
    if not os.path.exists(metrics_path):
        # 尝试从train_metrics中读取
        train_metrics_path = "models/metrics/train_metrics.json"
        if os.path.exists(train_metrics_path):
            with open(train_metrics_path) as f:
                data = json.load(f)
            return {"accuracy": data.get("best_val_acc", 0.0), "source": "train_metrics"}
        return {"accuracy": 0.0}

    with open(metrics_path) as f:
        return json.load(f)


def check_promotion_threshold(metrics: dict, thresholds: dict) -> tuple[bool, str]:
    """检查指标是否满足晋升到Production的阈值"""
    acc = metrics.get("accuracy", metrics.get("mAP50", 0.0))
    required_acc = thresholds.get("accuracy", 0.85)

    if acc < required_acc:
        reason = f"精度 {acc:.4f} < 阈值 {required_acc}"
        return False, reason

    # 延迟检查（如果有）
    latency = metrics.get("latency_ms")
    if latency is not None:
        max_latency = thresholds.get("latency_ms", 50)
        if latency > max_latency:
            reason = f"延迟 {latency}ms > 阈值 {max_latency}ms"
            return False, reason

    return True, "所有指标满足晋升阈值"


def register_model(cfg: dict, metrics: dict) -> dict:
    """注册模型到MLflow Registry"""
    registry_cfg = cfg.get("registry", {})
    model_name = registry_cfg.get("model_name", "visionops-detector")
    thresholds = registry_cfg.get("promotion_threshold", {})

    # 设置MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # 读取训练时的Run ID
    run_id = None
    run_id_path = "models/metrics/mlflow_run_id.txt"
    if os.path.exists(run_id_path):
        with open(run_id_path) as f:
            run_id = f.read().strip()

    rknn_path = "models/export/model.rknn"
    onnx_path = "models/export/model.onnx"

    # 如果没有run_id，创建新的run来记录artifact
    if not run_id:
        mlflow.set_experiment("visionops-model-registration")
        with mlflow.start_run(run_name=f"register-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
            run_id = run.info.run_id
            if os.path.exists(rknn_path):
                mlflow.log_artifact(rknn_path, artifact_path="rknn")
            if os.path.exists(onnx_path):
                mlflow.log_artifact(onnx_path, artifact_path="onnx")
            mlflow.log_metrics(metrics)

    # 注册模型版本
    model_uri = f"runs:/{run_id}/rknn"
    if not os.path.exists(rknn_path):
        model_uri = f"runs:/{run_id}/onnx"

    print(f"\n注册模型到 MLflow Registry...")
    print(f"  模型名称: {model_name}")
    print(f"  Run ID: {run_id}")

    # 创建或更新已注册模型
    try:
        client.get_registered_model(model_name)
        print(f"  已存在的注册模型: {model_name}")
    except Exception:
        client.create_registered_model(
            model_name,
            description=f"VisionOps视觉检测模型 | 目标平台: RK3588"
        )
        print(f"  创建新注册模型: {model_name}")

    # 创建模型版本（上传artifacts）
    mv = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/",
        run_id=run_id,
        description=f"自动注册 | 精度: {metrics.get('accuracy', 'N/A')} | {datetime.now().isoformat()}"
    )

    version = mv.version
    print(f"  创建版本: {version}")

    # 检查是否满足晋升阈值
    can_promote, reason = check_promotion_threshold(metrics, thresholds)

    if can_promote:
        # 晋升到Production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True  # 归档旧的Production版本
        )
        stage = "Production"
        print(f"  ✓ 模型晋升到 Production！原因: {reason}")
    else:
        # 标记为Staging，等待人工审核
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )
        stage = "Staging"
        print(f"  ⚠ 模型留在 Staging。原因: {reason}")

    # 添加标签
    client.set_model_version_tag(model_name, version, "platform", "rk3588")
    client.set_model_version_tag(model_name, version, "quantized", "int8")
    client.set_model_version_tag(model_name, version, "auto_registered", "true")

    result = {
        "status": "success",
        "model_name": model_name,
        "version": version,
        "stage": stage,
        "promoted": can_promote,
        "promotion_reason": reason,
        "run_id": run_id,
        "registered_at": datetime.now().isoformat(),
        "metrics": metrics
    }

    return result


def main():
    cfg = load_config("pipeline/configs/mlops.yaml")
    metrics = load_eval_metrics()

    print("=" * 60)
    print("模型注册到 MLflow Model Registry")
    print(f"当前指标: {json.dumps(metrics, indent=2)}")
    print("=" * 60)

    result = register_model(cfg, metrics)

    # 保存注册结果
    Path("models/metrics").mkdir(parents=True, exist_ok=True)
    with open("models/metrics/registry_result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"✓ 注册完成!")
    print(f"  模型: {result['model_name']} v{result['version']}")
    print(f"  阶段: {result['stage']}")
    print(f"  是否晋升Production: {result['promoted']}")
    print("=" * 60)

    if not result["promoted"]:
        print(f"\n⚠️  模型未自动晋升到Production，请人工审核后决定是否部署")
        sys.exit(0)  # 不报错，但需要人工干预


if __name__ == "__main__":
    main()
