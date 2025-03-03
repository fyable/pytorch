import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)

    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))

# class MySequential(nn.Module):
#     def __init__(self,*args):
#         super().__init__()
#         for block in args:
#             self._modules[block] = block
#
#     def forward(self,X):
#         for block in self._modules.values():
#             X = block(X)
#         return X
#
# X = torch.rand(2,20)
# net = MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
# print(net(X))

# x = torch.arange(4)
# y = torch.zeros(4)
# mydict = {'x':x,'y':y}
# torch.save(mydict,'x_file')
#
# mydict2 = torch.load("x_file",weights_only=True)
# print(mydict2)

net = MLP()
X = torch.randn(size=(2,20))
Y = net(X)
torch.save(net.state_dict(),'mlp.params')

clone = MLP()
clone.load_state_dict(torch.load("mlp.params",weights_only=True))
clone.eval()

Y_clone = clone(X)
print(Y_clone==Y)
