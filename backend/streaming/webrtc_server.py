"""
Minimal WebRTC signaling server using aiohttp + aiortc.

Weeks 1-2 scope: one file, one viewer, localhost. No STUN/TURN needed
since browser and server are on the same machine/network.

Run standalone for local testing:
    python -m backend.streaming.webrtc_server
"""
import asyncio
import logging
import os

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

from backend.streaming.file_video_track import FileVideoTrack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc_server")

VIDEO_PATH = os.environ.get("VIDEO_PATH", "backend/streaming/sample.mp4")

pcs = set()


async def offer(request):
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    video_track = FileVideoTrack(VIDEO_PATH)
    pc.addTrack(video_track)

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


async def on_shutdown(app):
    await asyncio.gather(*(pc.close() for pc in pcs))
    pcs.clear()


def create_app():
    app = web.Application()

    @web.middleware
    async def cors_middleware(request, handler):
        if request.method == "OPTIONS":
            resp = web.Response()
        else:
            resp = await handler(request)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return resp

    app.middlewares.append(cors_middleware)
    app.router.add_post("/offer", offer)
    app.router.add_options("/offer", lambda r: web.Response())
    app.on_shutdown.append(on_shutdown)
    return app


if __name__ == "__main__":
    web.run_app(create_app(), host="0.0.0.0", port=8080)
