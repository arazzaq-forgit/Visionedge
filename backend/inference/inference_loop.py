# backend/inference/inference_loop.py

import time
import numpy as np
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())


def get_mock_frame(imgsz=640):
    """
    Placeholder for a real decoded frame.
    Once Ritam's hardware decoder (backend/decoder/hardware_decoder.py)
    is ready, swap this out for actual frames from the video pipeline.
    """
    return torch.randn(1, 3, imgsz, imgsz, dtype=torch.float32, device="cuda")


def run_inference_loop(engine_path="models/yolov8s.engine", num_frames=100, imgsz=640):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    output_shape = tuple(context.get_tensor_shape(output_name))

    output_tensor = torch.empty(*output_shape, dtype=torch.float32, device="cuda")
    context.set_tensor_address(output_name, output_tensor.data_ptr())

    stream = torch.cuda.Stream()
    latencies = []

    for i in range(num_frames):
        frame = get_mock_frame(imgsz)
        context.set_tensor_address(input_name, frame.data_ptr())

        torch.cuda.synchronize()
        start = time.time()

        with torch.cuda.stream(stream):
            context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    print(f"Processed {num_frames} frames")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"P95 latency:     {p95_latency:.2f} ms")
    print(f"Effective FPS:   {1000 / avg_latency:.2f}")

    return avg_latency, p95_latency


if __name__ == "__main__":
    run_inference_loop()
