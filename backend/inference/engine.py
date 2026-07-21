# backend/inference/engine.py

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path: str, engine_path: str, fp16: bool = True):
    """
    Compile an ONNX model into a TensorRT engine for optimized inference.
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    print(f"Saved TensorRT engine to {engine_path}")

if __name__ == "__main__":
    build_engine("yolov10.onnx", "yolov10.engine")