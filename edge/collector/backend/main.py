#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import STATIC_DIR, UI_VERSION
from backend.routers.collector import router as collector_router
from backend.routers.settings import router as settings_router
from backend.routers.cpp_inference import router as cpp_inference_router
from backend.services.camera import backend_camera_enabled, camera_service
from backend.services.storage import ensure_data_root


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_root()
    if backend_camera_enabled():
        camera_service.start()
    yield
    if backend_camera_enabled():
        camera_service.stop()


app = FastAPI(
    title="VisionOps Edge Collector UI",
    description="Calibration + collection + validation + production monitor local functional UI.",
    version=UI_VERSION,
    lifespan=lifespan,
)

app.include_router(collector_router)
app.include_router(settings_router)
app.include_router(cpp_inference_router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
