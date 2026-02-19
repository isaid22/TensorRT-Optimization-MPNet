import tensorrt as trt
import os

def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    # Create network definition with explicit batch
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Create ONNX parser
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Create build configuration
    config = builder.create_builder_config()
    
    # Set memory pool limit (e.g. 4GB for workspace)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    
    # Set FP16 mode
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Check if ONNX file exists
    if not os.path.exists(onnx_file_path):
        print(f"ONNX file not found: {onnx_file_path}")
        return

    # Parse ONNX model
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # Define optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    
    # Set input shapes based on trt-optimize.sh arguments
    # minShapes=input_ids:1x1,attention_mask:1x1
    # optShapes=input_ids:1x128,attention_mask:1x128
    # maxShapes=input_ids:1x512,attention_mask:1x512
    
    input_ids = network.get_input(0)
    attention_mask = network.get_input(1)
    
    # Assuming standard names 'input_ids' and 'attention_mask'. 
    # If the names in ONNX are different, we might need to look them up.
    # The trtexec command used explicit names so we assume they match.
    
    profile.set_shape("input_ids", (1, 1), (1, 128), (1, 512))
    profile.set_shape("attention_mask", (1, 1), (1, 128), (1, 512))
    
    config.add_optimization_profile(profile)

    # Build engine
    print(f"Building TensorRT engine from {onnx_file_path}...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine:
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print(f"Engine saved to {engine_file_path}")
    else:
        print("Failed to build engine.")

if __name__ == "__main__":
    onnx_path = "onnx_model/model.onnx"
    engine_path = "mpnet.engine"
    build_engine(onnx_path, engine_path)
