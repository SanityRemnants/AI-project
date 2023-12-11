from torch import nn
class AIcolider(nn.Module):
    def __init__(self):
        super(AIcolider, self).__init__()
        self.hidden1 = nn.Linear(6,512)
        self.activation1 = nn.Tanh()
        self.hidden2 = nn.Linear(512, 256)
        self.activation2 = nn.Tanh()
        self.hidden3 = nn.Linear(256, 128)
        self.activation3 = nn.ReLU()
        self.output = nn.Linear(128,4)
        pass
    def forward(self,x):
        x = self.activation1(self.hidden1(x))
        x = self.activation2(self.hidden2(x))
        x = self.activation3(self.hidden3(x))
        x = self.output(x)
        return x