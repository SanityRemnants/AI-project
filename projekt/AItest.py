from AIcollider import AIcolider
import torch
aicollider = AIcolider()
aicollider.load_state_dict(torch.load("model_best"))
a = aicollider(torch.FloatTensor([0.997588087,-0.069411884,-0.048088927,-0.038272075,0.116477087,0.206267044])).tolist()
print(a)