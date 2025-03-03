import torch
from torch import nn

def try_gpu(i=0):
    if torch.cuda.device_count()>=i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    devices=[
        torch.device(f'cuda:{i}')for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device('cpu')]

X=torch.tensor([1.,2.,3.],device=try_gpu())
net=nn.Sequential(nn.Linear(3,1))
net=net.to(device=try_gpu())
print(net[0].weight.data.device)