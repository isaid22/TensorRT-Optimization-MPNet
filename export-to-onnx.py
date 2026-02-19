from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = "sentence-transformers/all-mpnet-base-v2"

# This command downloads, converts to ONNX, and saves to 'onnx_model/'
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained("onnx_model")
tokenizer.save_pretrained("onnx_model")

print("Export successful!")