"""
FileVideoTrack: a simple aiortc VideoStreamTrack that reads frames
from a local video file (mp4/avi/etc.) and loops forever.

Weeks 1-2 goal: get an UNEDITED file streaming to the browser.
No GPU decode yet, no CuPy, no zero-copy -- that plugs in during
weeks 3-4 once Ritam's NVDEC pipeline is ready to hand off frames.
"""
import cv2
from av import VideoFrame
from aiortc import VideoStreamTrack


class FileVideoTrack(VideoStreamTrack):
    """Streams frames from a local video file on loop."""

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {file_path}")

    def _read_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            # loop back to the start of the file
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame even after rewind")
        return frame

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame_bgr = self._read_frame()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame
