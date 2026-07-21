# backend/inference/model_compile.py

import torch

def export_to_onnx(model_path: str, output_path: str):
    """
    Load a pre-trained YOLO model and export it to ONNX format.
    """
    model = torch.load(model_path)
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    torch.onnx.export(model, dummy_input, output_path, opset_version=12)
    print(f"Exported ONNX model to {output_path}")

if __name__ == "__main__":
    export_to_onnx("yolov10.pt", "yolov10.onnx")