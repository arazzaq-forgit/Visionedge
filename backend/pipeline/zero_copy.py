# backend/pipeline/zero_copy.py

import time
import cupy as cp
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())


def get_mock_gpu_frame(imgsz=640):
    """
    Simulates a frame that's already decoded directly onto the GPU
    (as it will be once Ritam's NVDEC decoder hands off frames here).
    Using CuPy keeps this entirely in VRAM -- no CPU/numpy involved.
    """
    return cp.random.randn(1, 3, imgsz, imgsz, dtype=cp.float32)


def run_zero_copy_pipeline(engine_path="models/yolov8s.engine", num_frames=100, imgsz=640):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    output_shape = tuple(context.get_tensor_shape(output_name))

    # Output buffer also stays on GPU via CuPy -- no CPU roundtrip
    output_buffer = cp.empty(output_shape, dtype=cp.float32)

    context.set_tensor_address(output_name, output_buffer.data.ptr)

    stream = cp.cuda.Stream()
    latencies = []

    # Warm-up (first few calls include CUDA kernel compilation/caching overhead)
    for _ in range(10):
        frame = get_mock_gpu_frame(imgsz)
        context.set_tensor_address(input_name, frame.data.ptr)
        with stream:
            context.execute_async_v3(stream_handle=stream.ptr)
        stream.synchronize()

    for _ in range(num_frames):
        frame = get_mock_gpu_frame(imgsz)  # already on GPU, zero-copy
        context.set_tensor_address(input_name, frame.data.ptr)

        cp.cuda.Stream.null.synchronize()
        start = time.time()

        with stream:
            context.execute_async_v3(stream_handle=stream.ptr)
        stream.synchronize()

        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    print(f"[Zero-Copy Pipeline] Processed {num_frames} frames")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"P95 latency:     {p95_latency:.2f} ms")
    print(f"Effective FPS:   {1000 / avg_latency:.2f}")

    return avg_latency, p95_latency


if __name__ == "__main__":
    run_zero_copy_pipeline()