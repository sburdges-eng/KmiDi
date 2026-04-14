import torch
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained("labhamlet/wavjepa-base", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("labhamlet/wavjepa-base", trust_remote_code=True)
import numpy as np
audio_16k = np.zeros(16000, dtype=np.float32)
inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
print("Input shape:", inputs.input_values.shape)
out = model(inputs.input_values)
print("Type of out:", type(out))
if isinstance(out, tuple):
    print("Tuple len:", len(out))
elif hasattr(out, "__dict__"):
    print("Keys:", out.__dict__.keys())
if hasattr(out, "last_hidden_state"):
    print("last_hidden_state:", out.last_hidden_state.shape)
else:
    print("out:", out)
