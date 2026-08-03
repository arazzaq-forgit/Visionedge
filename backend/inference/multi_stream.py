# backend/inference/multi_stream.py
"""
Week 4: Multi-Stream Orchestration.
Manages several independent inference "streams" concurrently on a single
GPU using asyncio, each running against the same TensorRT engine.
"""

import asyncio
import time
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())


class InferenceStream:
    """
    Represents one independent video stream's inference context.
    Each stream gets its own TensorRT execution context and CUDA stream,
    so multiple streams can be in-flight concurrently on the same GPU
    (the engine's weights are shared; only per-stream state is separate).
    """

    def __init__(self, stream_id: str, engine, imgsz: int = 640):
        self.stream_id = stream_id
        self.context = engine.create_execution_context()
        self.imgsz = imgsz
        self.cuda_stream = torch.cuda.Stream()
        self.frame_count = 0
        self.total_latency_ms = 0.0

        self.input_name = engine.get_tensor_name(0)
        self.output_name = engine.get_tensor_name(1)
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        self.output_tensor = torch.empty(*output_shape, dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(self.output_name, self.output_tensor.data_ptr())

    async def process_frame(self):
        """Runs one inference pass for this stream (mock frame for now)."""
        frame = torch.randn(1, 3, self.imgsz, self.imgsz, dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(self.input_name, frame.data_ptr())

        start = time.time()
        with torch.cuda.stream(self.cuda_stream):
            self.context.execute_async_v3(stream_handle=self.cuda_stream.cuda_stream)

        # Yield control back to the event loop while the GPU works,
        # instead of blocking -- this is what lets multiple streams
        # genuinely overlap on the same GPU.
        while not self.cuda_stream.query():
            await asyncio.sleep(0)

        latency_ms = (time.time() - start) * 1000
        self.frame_count += 1
        self.total_latency_ms += latency_ms
        return latency_ms

    async def run(self, num_frames: int):
        for _ in range(num_frames):
            await self.process_frame()

    @property
    def avg_latency_ms(self):
        return self.total_latency_ms / self.frame_count if self.frame_count else 0.0


async def run_multi_stream(engine_path="models/yolov8s.engine", num_streams=4, frames_per_stream=50):
    engine = load_engine(engine_path)
    streams = [InferenceStream(f"stream-{i}", engine) for i in range(num_streams)]

    print(f"Starting {num_streams} concurrent streams, {frames_per_stream} frames each...")
    start = time.time()

    await asyncio.gather(*(s.run(frames_per_stream) for s in streams))

    elapsed = time.time() - start
    total_frames = num_streams * frames_per_stream
    aggregate_fps = total_frames / elapsed

    print(f"\n=== Multi-Stream Results ===")
    print(f"Streams:            {num_streams}")
    print(f"Total frames:       {total_frames}")
    print(f"Wall-clock time:    {elapsed:.2f}s")
    print(f"Aggregate FPS:      {aggregate_fps:.2f}")
    for s in streams:
        print(f"  {s.stream_id}: avg latency {s.avg_latency_ms:.2f} ms over {s.frame_count} frames")

    return aggregate_fps


if __name__ == "__main__":
    asyncio.run(run_multi_stream())