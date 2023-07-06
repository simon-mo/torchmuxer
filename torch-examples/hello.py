import torch
from fijit_py import Fijit
import time

# f = Fijit()
# f.run()

tensor = torch.zeros(2).to("cuda")
for _ in range(10):
    tensor.add_(1)
    time.sleep(0.5)
tensor.to("cpu")
