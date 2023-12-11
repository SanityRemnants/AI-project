from torch import nn
class AImover(nn.Module):
    def __init__(self):
        super(AImover, self).__init__()
        self.hidden1 = nn.Linear(1,4)
        self.ReLU = nn.ReLU()
        self.output = nn.Linear(4,1)
        pass
    def forward(self,x):
        x = self.ReLU(self.hidden1(x))
        x = self.output(x)
        return x