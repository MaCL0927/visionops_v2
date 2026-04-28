#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compatibility entrypoint for VisionOps Edge Collector."""

import uvicorn
from backend.main import app

if __name__ == "__main__":
    # 板端运行不要开启 reload，否则会多一个 watcher/reloader 进程并增加负载。
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8090, reload=False, workers=1)
