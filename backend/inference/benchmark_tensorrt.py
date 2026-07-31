# backend/inference/benchmark_tensorrt.py

import time
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())

def benchmark_tensorrt(engine_path="models/yolov8s.engine", num_runs=100, imgsz=640):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    input_shape = (1, 3, imgsz, imgsz)
    output_shape = tuple(context.get_tensor_shape(output_name))

    # Use PyTorch CUDA tensors as GPU buffers (no pycuda needed)
    input_tensor = torch.randn(*input_shape, dtype=torch.float32, device="cuda")
    output_tensor = torch.empty(*output_shape, dtype=torch.float32, device="cuda")

    context.set_tensor_address(input_name, input_tensor.data_ptr())
    context.set_tensor_address(output_name, output_tensor.data_ptr())

    stream = torch.cuda.Stream()

    # Warm-up
    for _ in range(10):
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_runs):
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

    elapsed = time.time() - start
    fps = num_runs / elapsed
    print(f"[TensorRT] {num_runs} runs in {elapsed:.2f}s -> {fps:.2f} FPS")
    return fps

if __name__ == "__main__":
    benchmark_tensorrt()
