import math

from AIcollider import AIcolider
from AImover import AImover
import torch
aicollider = AImover()
aicollider.load_state_dict(torch.load("model2_best"))
inputs = [1]
print(aicollider(torch.FloatTensor(inputs)))
print(aicollider.hidden1.weight)

