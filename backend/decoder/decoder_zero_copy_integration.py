# backend/decoder/decoder_zero_copy_integration.py
"""
Weeks 3-4: Connects the hardware decoder's real video frames into the
zero-copy CuPy/TensorRT pipeline, replacing mock frames with actual
decoded video -- and checks for VRAM leaks across a longer run.
"""

import time
import cupy as cp
import torch
import tensorrt as trt

from backend.decoder.hardware_decoder import decode_stream_gpu

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())


def frame_to_gpu_tensor(frame_bgr, imgsz=640):
    """
    Takes a decoded BGR frame (numpy, from PyAV) and moves it to GPU
    as a CuPy array, resized/reshaped to match the model's input shape.
    """
    import cv2
    resized = cv2.resize(frame_bgr, (imgsz, imgsz))
    normalized = resized.astype("float32") / 255.0
    chw = normalized.transpose(2, 0, 1)  # HWC -> CHW
    batched = chw[None, ...]  # add batch dim
    return cp.asarray(batched)  # upload to GPU as CuPy array


def run_decoder_to_pipeline(video_path, engine_path="models/yolov8s.engine", imgsz=640, max_frames=100):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    output_shape = tuple(context.get_tensor_shape(output_name))
    output_buffer = cp.empty(output_shape, dtype=cp.float32)
    context.set_tensor_address(output_name, output_buffer.data.ptr)

    stream = cp.cuda.Stream()
    latencies = []
    vram_readings = []

    frame_count = 0
    for frame_bgr in decode_stream_gpu(video_path):
        gpu_frame = frame_to_gpu_tensor(frame_bgr, imgsz)
        context.set_tensor_address(input_name, gpu_frame.data.ptr)

        cp.cuda.Stream.null.synchronize()
        start = time.time()

        with stream:
            context.execute_async_v3(stream_handle=stream.ptr)
        stream.synchronize()

        latencies.append((time.time() - start) * 1000)

        allocated_mb = cp.get_default_memory_pool().used_bytes() / (1024 ** 2)
        vram_readings.append(allocated_mb)

        frame_count += 1
        if frame_count >= max_frames:
            break

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    vram_start = vram_readings[0] if vram_readings else 0
    vram_end = vram_readings[-1] if vram_readings else 0
    vram_growth = vram_end - vram_start

    print(f"\n=== Decoder -> Zero-Copy Pipeline Integration ===")
    print(f"Frames processed:   {frame_count}")
    print(f"Average latency:    {avg_latency:.2f} ms")
    print(f"VRAM at start:      {vram_start:.2f} MB")
    print(f"VRAM at end:        {vram_end:.2f} MB")
    print(f"VRAM growth:        {vram_growth:.2f} MB")
    print(f"Leak check:         {'POTENTIAL LEAK' if vram_growth > 50 else 'OK - no significant growth'}")

    return avg_latency, vram_growth


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "backend/streaming/sample.mp4"
    run_decoder_to_pipeline(video)