import numpy as np
import os

# Create a directory for calibration data
if not os.path.exists("calib_data"):
    os.makedirs("calib_data")

# Generate dummy input data
dummy_input_ids = np.random.randint(0, 30522, (10, 128), dtype=np.int64)
dummy_attention_mask = np.ones((10, 128), dtype=np.int64)

# Save to files
np.save("calib_data/input_ids.npy", dummy_input_ids)
np.save("calib_data/attention_mask.npy", dummy_attention_mask)

print("Generated calibration data in calib_data/")
