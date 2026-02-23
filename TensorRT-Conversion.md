# TensorRT-Optimization-MPNet

For INT8 Quantization, see [README.md](./README.md)

This repo contains works for using TensorRT to optimize MPNet in FP16 data type. This is a minimal example to achieve the following:

1. Export an encoder model to ONNX format.
2. Optimize the ONNX format model using TensorRT, with datatype FP16.
3. Run an inference with one pass sample text.

The following are explanations for each step. This example uses AWS `g5.4xlarge` instance with Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Ubuntu 24.04) ami-0f3d7b789119ccbfa. 

## ONNX Conversion

The work to convert a native model to ONNX format is shown in [`export-to-onnx.py`](export-to-onnx.py). This script should be executed very quicky. The result is a folder `onnx_model`. This folder contains the following:

```
-rw-rw-r-- 1 ubuntu ubuntu       529 Feb 19 00:59 config.json
-rw-rw-r-- 1 ubuntu ubuntu 435760242 Feb 19 00:59 model.onnx
-rw-rw-r-- 1 ubuntu ubuntu      1615 Feb 19 00:59 tokenizer_config.json
-rw-rw-r-- 1 ubuntu ubuntu       964 Feb 19 00:59 special_tokens_map.json
-rw-rw-r-- 1 ubuntu ubuntu    231536 Feb 19 00:59 vocab.txt
-rw-rw-r-- 1 ubuntu ubuntu    710944 Feb 19 00:59 tokenizer.json
```

here, the model is converted to onnx format as shown in `model.onnx`. The model configuration and hyperparameters are shown in `config.json`. 

## TensorRT Optimization
This step is done with [`build_engine.py`](./build_engine.py). Notice the lines

```
# Format: (Batch, Length)
# (Min Batch, Min Len), (Opt Batch, Opt Len), (Max Batch, Max Len)
profile.set_shape("input_ids", (1, 1), (1, 128), (1, 512))
profile.set_shape("attention_mask", (1, 1), (1, 128), (1, 512))
```

The first dimension is always 1 in this case. This is because the batch size of choice here is 1, regardless of token lengths, and I want to optimize this model to handle this batch size and token length. 

If I have a set of preferred batch sizes and token lengths in mind, I will modify above to:

```
profile.set_shape("input_ids", (1, 1), (16, 128), (32, 512))
profile.set_shape("attention_mask", (1, 1), (16, 128), (32, 512))
```

Now the model will be optimized by TensorRT to handle these batch sizes. 

You may see 

```
Building TensorRT engine from onnx_model/model.onnx...
Engine saved to mpnet.engine
``` 

on the terminal. Notice the size of pf mpnet.engine. It should be half of model.onnx, because of FP16 datatype.

## Inference
This step is done with [`inference-trt.py`](inference-trt.py). This script performs the following steps:

1. Loads the Engine: Deserializes mpnet.engine into a TensorRT runtime.
2. Prepares Input: Uses the Hugging Face tokenizer to convert a text string ("This is an example sentence.") into input_ids and attention_mask.
3. Allocates Memory: Allocates pinned memory on the Host (CPU) and standard memory on the Device (GPU). 
4. Executes: Copies inputs to GPU, runs the TensorRT engine execution context, and copies the result back to CPU.
Result: It prints the final embedding tensor.

[!IMPORTANT]
Memory lifecycle is an important nuace to keep in mind. In the code:

```
for i in range(self.engine.num_io_tensors):
    # ...
    device_mem = cuda.mem_alloc(host_mem.nbytes)  # <--- created here
    
    # We pass the memory ADDRESS (integer) to TensorRT
    self.context.set_tensor_address(tensor_name, int(device_mem)) 
    
    # End of loop iteration
```

device_mem is a local Python object (of type pycuda.driver.DeviceAllocation). When the loop iteration finishes, device_mem goes out of scope. Since nothing else is holding onto this object, Python's garbage collector destroys it.
The Consequence: When device_mem is destroyed, PyCUDA automatically frees the GPU memory. Even though we gave the address to TensorRT, that address becomes invalid immediately. When execute_async_v3 runs later, it tries to access freed memory, causing a crash or garbage output.

The Fix
I added a list to keep these objects alive explicitly.

Line 20: I initialized a list to store the references.
```
self.allocations = [] # Keep references to prevent GC
```

Line 32: Inside the loop, I saved the device_mem object into that list.
```
device_mem = cuda.mem_alloc(host_mem.nbytes)
self.allocations.append(device_mem) # Store reference !!!
```

Line 51: After execution is finished, I clear the list, allowing the memory to be freed normally.
```
self.allocations = []
```

This ensures the GPU memory remains allocated for the entire duration of the inference call.

### Run setup

The for loop (lines 26–51) runs entirely on the CPU and is purely for setup. It iterates 3 times (for input_ids, attention_mask, and the output tensor) to do the following:

1. Allocate memory (cuda.mem_alloc).
2. Copy data from CPU to GPU (cuda.memcpy_htod_async).
3. Tell TensorRT where that memory is (set_tensor_address).
Because this is Python code running on the CPU, it must run sequentially.

The actual GPU execution happens at Line 53, completely outside the loop:
```
self.context.execute_async_v3(self.stream.handle)
```

This is the single command that launches the entire neural network on the GPU. Once this line triggers, the GPU takes over and runs all the layers of the model, utilizing its massive parallelism.

In summary:

The Loop: Serial Setup (CPU) - "Here is the data."
Line 53: Parallel Execution (GPU) - "Run the model."


### Execution

The actual GPU execution happens at Line 53, completely outside the loop:

```
self.context.execute_async_v3(self.stream.handle)
```
This is the single command that launches the entire neural network on the GPU. Once this line triggers, the GPU takes over and runs all the layers of the model, utilizing its massive parallelism.

Basically: 
The Loop: Serial Setup (CPU) - "Here is the data."
Line 53: Parallel Execution (GPU) - "Run the model."

`execute_async_v3` takes no data arguments. That is because the data was already registered inside the `context` object before that line was called.

Points to look at in inference-trt.py:

The Handshake (Line 48):

```
self.context.set_tensor_address(tensor_name, int(device_mem))
```

This is the crucial line. You are telling the TensorRT context: "Hey, when you run, if you need the tensor named input_ids, look at GPU memory address 0x12345678."

The Trigger (Line 53):
```
self.context.execute_async_v3(self.stream.handle)
```

When you call this, TensorRT doesn't need arguments because it already has the map of names to memory addresses. It just looks up the address you registered in step 1 and starts reading from there.



The loop `for i in range(self.engine.num_io_tensors)` iterates over ALL tensors that go in or out of the model.

For this MPNet model, `self.engine.num_io_tensors` is likely 3. The loop runs 3 times:

1. Iteration 1 (Input): Tensor is "input_ids".
get_tensor_mode returns INPUT.
Action: Copies the text data to GPU.
2. Iteration 2 (Input): Tensor is "attention_mask".
get_tensor_mode returns INPUT.
Action: Copies the mask data to GPU.
3. Iteration 3 (Output): Tensor is "last_hidden_state" (or similar).
get_tensor_mode returns OUTPUT.
Action: Enters the else block.
It doesn't copy anything to the GPU here. Instead, it saves the memory pointers (output_mem, d_output) so that after the model runs, it knows where to copy the results back from.
So, the else block is guaranteed to run exactly once (for this model) simply because the model has an output layer.

Below is expected result of inference:
```
/home/ubuntu/ai_venv/lib/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Inference output (embedding):
[[[ 0.05776978 -0.15673828 -0.01065826 ... -0.00502777  0.15405273
   -0.019104  ]
  [ 0.01416016 -0.22570801 -0.09771729 ... -0.04104614  0.04962158
   -0.01191711]
  [-0.02050781 -0.41845703 -0.07580566 ... -0.11193848  0.11920166
    0.05178833]
  ...
  [ 0.05697632  0.01589966 -0.00187302 ...  0.046875    0.10852051
   -0.02911377]
  [ 0.05697632  0.01589966 -0.00187302 ...  0.046875    0.10852051
   -0.02911377]
  [ 0.05697632  0.01589966 -0.00187302 ...  0.046875    0.10852051
   -0.02911377]]]
```

The terminal prints out the output embedding vector.