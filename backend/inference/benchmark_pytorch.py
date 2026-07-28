# backend/inference/benchmark_pytorch.py

import time
import torch
from ultralytics import YOLO

def benchmark_pytorch(model_path="models/yolov8s.pt", num_runs=100, imgsz=640):
    yolo = YOLO(model_path)
    model = yolo.model.to("cuda").eval().half()  # raw nn.Module, FP16 to match TensorRT

    dummy_input = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float16, device="cuda")

    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(dummy_input)

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(num_runs):
            _ = model(dummy_input)

        torch.cuda.synchronize()
        elapsed = time.time() - start

    fps = num_runs / elapsed
    print(f"[PyTorch raw forward] {num_runs} runs in {elapsed:.2f}s -> {fps:.2f} FPS")
    return fps

if __name__ == "__main__":
    benchmark_pytorch()
