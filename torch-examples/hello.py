import torch
from  fijit_py import Fijit

# f = Fijit()
# f.run()

tensor = torch.zeros(2).to("cuda")
tensor.add_(1)
tensor.to("cpu")
