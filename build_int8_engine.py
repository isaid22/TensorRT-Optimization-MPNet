import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Simple calibrator for demonstration
class MyCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, data_dir, cache_file):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.data_dir = data_dir
        self.cache_file = cache_file
        self.batch_size = 1
        self.input_names = ["input_ids", "attention_mask"]
        self.input_files = {
            "input_ids": np.load(os.path.join(self.data_dir, "input_ids.npy")),
            "attention_mask": np.load(os.path.join(self.data_dir, "attention_mask.npy"))
        }
        self.current_index = 0
        self.device_inputs = {}
        for name in self.input_names:
            self.device_inputs[name] = cuda.mem_alloc(self.input_files[name][0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > self.input_files["input_ids"].shape[0]:
            return None
        
        batch_data = {}
        for name in self.input_names:
            batch_data[name] = self.input_files[name][self.current_index:self.current_index+self.batch_size]

        for name in names:
            cuda.memcpy_htod(self.device_inputs[name], batch_data[name])
        
        self.current_index += self.batch_size
        return [int(self.device_inputs[name]) for name in names]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def main():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open("/home/ubuntu/Model-Optimization/onnx_model/model.onnx", "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)

    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1, 1), (1, 128), (1, 512))
    profile.set_shape("attention_mask", (1, 1), (1, 128), (1, 512))
    config.add_optimization_profile(profile)
    
    calibrator = MyCalibrator("calib_data", "calibration.cache")
    config.int8_calibrator = calibrator
    
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        print("Failed to build engine")
        return

    with open("mpnet-int8.engine", "wb") as f:
        f.write(engine)
    print("Successfully created INT8 engine.")

if __name__ == "__main__":
    main()
