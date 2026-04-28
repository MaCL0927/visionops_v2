from __future__ import annotations

from pipeline.core.config import get_task_type


def main() -> None:
    task_type = get_task_type()
    if task_type == "detection":
        from pipeline.tasks.detection.evaluate import main as run
    elif task_type == "classification":
        from pipeline.tasks.classification.evaluate import main as run
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")
    run()


if __name__ == "__main__":
    main()
