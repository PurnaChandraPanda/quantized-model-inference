import sys

import torch


print(f"Python:               {sys.version.split()[0]}")

print(f"PyTorch:              {torch.__version__}")
print(f"PyTorch CUDA build:   {torch.version.cuda}")
print(f"CUDA available:       {torch.cuda.is_available()}")
print(f"Visible GPU count:    {torch.cuda.device_count()}")
