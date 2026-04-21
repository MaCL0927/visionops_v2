FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    openssh-client curl git \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖（分层缓存，依赖变化才重建）
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制代码
COPY . .

# 创建必要目录
RUN mkdir -p \
    models/checkpoints \
    models/export \
    models/metrics \
    data/raw \
    data/processed \
    logs/retrain

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
