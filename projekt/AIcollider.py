from torch import nn
class AIcolider(nn.Module):
    def __init__(self):
        super(AIcolider, self).__init__()
        self.hidden1 = nn.Linear(6,64)
        self.activation1 = nn.Tanh()
        self.hidden2 = nn.Linear(64, 64)
        self.activation2 = nn.ReLU()
        self.hidden3 = nn.Linear(64, 64)
        self.activation3 = nn.ReLU()
        self.output = nn.Linear(64,4)
        pass
    def forward(self,x):
        x = self.activation1(self.hidden1(x))
        x = self.activation2(self.hidden2(x))
        x = self.activation3(self.hidden3(x))
        x = self.output(x)
        return x