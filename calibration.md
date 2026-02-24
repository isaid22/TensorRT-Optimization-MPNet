# Calibration for Quantization

Calibration is the process of finding the actual range of floating-point values that flow through the model during inference. By feeding the model a small, representative sample of the data, TensorRT can observe the real-world minimum and maximum values for each layer's activations. This allows it to create a much more precise and tailored mapping to the INT8 range, which is the key to preserving accuracy.

In here, two main components were created specifically for this purpose, as defined in [build_int8_engine.py](./build_int8_engine.py):

The `MyCalibrator` Class: This is a custom Python class you wrote that acts as a data provider for TensorRT. It inherits from TensorRT's `IInt8MinMaxCalibrator` and its job is to feed batches of calibration data to the TensorRT builder.

The Calibration Dataset (calib_data directory): This directory holds the actual data used for the calibration process. Based on  `build_int8_engine.py` script, it contains at least two files:

```python
input_ids.npy
attention_mask.npy
```

These files contain pre-processed, real-world examples of inputs that MPNet model would expect to see. This dataset should be a representative subset of the validation or training data.

## How does calibration work

When running python [build_int8_engine.py](./build_int8_engine.py), here is the sequence of events for calibration:

1. Builder Detects INT8 Mode: The TensorRT builder sees that has set the INT8 flag and that has provided a calibrator object (config.int8_calibrator = calibrator).

2. TensorRT Asks for Data: The builder now needs to "see" the data flowing through the network. It turns to the MyCalibrator object and starts interacting with it.

3. The get_batch Method is Called: TensorRT repeatedly calls the get_batch() method of MyCalibrator instance.

* Each time it's called, the code loads one batch of data from the .npy files.
* It then copies this batch from the CPU's memory to the GPU's memory.
* Finally, it returns the GPU memory addresses of the input tensors to TensorRT.

4. TensorRT Observes the Activations: With the data for a batch on the GPU, TensorRT performs a forward pass through the network. As the data flows through each layer, TensorRT records the minimum and maximum activation values it observes. It does this for every batch the calibrator provides.

5. Calculating the Scaling Factors: After running through all the calibration batches, TensorRT has a good statistical picture of the dynamic range of activations for every tensor in the graph. It uses this information to calculate a "scaling factor" for each one. This scaling factor is the precise value needed to map the observed floating-point range to the limited INT8 range.

6. Caching the Results: The calibration process can be slow. To avoid repeating it every time, MyCalibrator implements write_calibration_cache(). After the process is complete, TensorRT passes the calibration results (the calculated scaling factors) to this method, which saves them to the file calibration.cache. The next time you run the script, read_calibration_cache() will find this file and load it, allowing TensorRT to skip the entire calibration process.

In summary, MyCalibrator class and the calib_data directory work together to provide a "live demonstration" for TensorRT, allowing it to learn the unique characteristics of the model and create an optimized, accurate INT8 engine.