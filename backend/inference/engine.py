# backend/inference/engine.py

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path: str, engine_path: str):
    """
    Compile an ONNX model into a TensorRT engine for optimized inference.
    (FP16 tuning to be revisited later — TensorRT 11's precision API
    differs significantly from older tutorials/docs.)
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Saved TensorRT engine to {engine_path}")

if __name__ == "__main__":
    build_engine("models/yolov8n.onnx", "models/yolov8n.engine")