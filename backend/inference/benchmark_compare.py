# backend/inference/benchmark_compare.py

from benchmark_pytorch import benchmark_pytorch
from benchmark_tensorrt import benchmark_tensorrt

if __name__ == "__main__":
    print("Running PyTorch benchmark...")
    pytorch_fps = benchmark_pytorch()

    print("\nRunning TensorRT benchmark...")
    trt_fps = benchmark_tensorrt()

    speedup = trt_fps / pytorch_fps
    print(f"\n=== RESULTS ===")
    print(f"PyTorch FPS:  {pytorch_fps:.2f}")
    print(f"TensorRT FPS: {trt_fps:.2f}")
    print(f"Speedup:      {speedup:.2f}x")
    print(f"Meets 3x target: {'YES' if speedup >= 3 else 'NO'}")