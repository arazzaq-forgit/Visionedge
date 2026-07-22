# backend/inference/model_compile.py

from ultralytics import YOLO

def export_to_onnx(model_name: str = "yolov8n.pt", opset: int = 12):
    """
    Load a pre-trained YOLO model and export it to ONNX format.
    Uses Ultralytics' built-in export pipeline (handles PyTorch -> ONNX
    conversion, dynamic axes, and opset compatibility automatically).
    """
    model = YOLO(model_name)
    export_path = model.export(format="onnx", opset=opset)
    print(f"Exported ONNX model to {export_path}")
    return export_path

if __name__ == "__main__":
    export_to_onnx("models/yolov8n.pt")