# backend/streaming/metrics.py
"""
Lightweight metrics endpoint for the telemetry dashboard.
Uses torch.cuda for GPU memory stats; FPS/decoder utilization
are placeholders until wired into the real pipeline (Weeks 3-4).
"""
import torch
from aiohttp import web

_current_fps = {"value": 0.0}
_decoder_util = {"value": 0.0}


def update_fps(value: float):
    _current_fps["value"] = value


def update_decoder_util(value: float):
    _decoder_util["value"] = value


async def metrics_handler(request):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0)
        total = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_percent = (allocated / total) * 100
    else:
        gpu_memory_percent = 0.0

    return web.json_response({
        "fps": _current_fps["value"],
        "gpuMemoryPercent": round(gpu_memory_percent, 2),
        "decoderUtilization": _decoder_util["value"],
    })


def register_metrics_route(app):
    app.router.add_get("/metrics", metrics_handler)
