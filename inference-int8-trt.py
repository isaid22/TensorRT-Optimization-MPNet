import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from transformers import AutoTokenizer

class TRTInference:
    def __init__(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    def infer(self, input_ids, attention_mask):
        # Set concrete shapes so dynamic dims (-1) are resolved before allocation.
        self.context.set_input_shape("input_ids", input_ids.shape)
        self.context.set_input_shape("attention_mask", attention_mask.shape)

        self.allocations = [] # Keep references to prevent GC
        output_mem = None
        output_shape = None
        d_output = None

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            # Use max shape if dynamic? Ideally use current shape
            shape = self.context.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            size = trt.volume(shape)
            
            # Host memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            # Device memory
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.allocations.append(device_mem) # Store reference

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                if "input_ids" in tensor_name:
                    np.copyto(host_mem, input_ids.ravel())
                elif "attention_mask" in tensor_name:
                    np.copyto(host_mem, attention_mask.ravel())
                cuda.memcpy_htod_async(device_mem, host_mem, self.stream)
            else:
                output_mem = host_mem
                output_shape = shape
                d_output = device_mem

            # Required for TRT v3 execution API.
            self.context.set_tensor_address(tensor_name, int(device_mem))

        self.context.execute_async_v3(self.stream.handle)

        cuda.memcpy_dtoh_async(output_mem, d_output, self.stream)
        self.stream.synchronize()
        
        # Clean up allocations
        self.allocations = []

        output_mem.shape = tuple(output_shape)
        return output_mem


def main():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    trt_inference = TRTInference("mpnet-int8.engine")

    text = "This is an example sentence."
    inputs = tokenizer(text, return_tensors="np", padding="max_length", max_length=128, truncation=True)
    input_ids = inputs["input_ids"].astype(np.int32)
    attention_mask = inputs["attention_mask"].astype(np.int32)

    output = trt_inference.infer(input_ids, attention_mask)

    print("Inference output (embedding):")
    print(output)

if __name__ == "__main__":
    main()
