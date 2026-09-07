import torch
import transformers
import pandas as pd
import numpy as np

print("=" * 60)

print("PyTorch Version      :", torch.__version__)

print("Transformers Version :", transformers.__version__)

print("NumPy Version        :", np.__version__)

print("Pandas Version       :", pd.__version__)

print("=" * 60)

print("CUDA Available :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

print("=" * 60)

print("Everything Installed Successfully!")