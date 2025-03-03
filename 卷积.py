import torch
from torch import nn
from d2l import torch as d2l

class Conv3D(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight=nn.Parameter(torch.rand(kernel_size))
        self.bias=nn.Parameter(torch.zeros(1))

    def forward(self,x):
        return corr2d(x,self.weight)+self.bias

def corr2d(X,K):
    h,w=K.shape
    Y=torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(X[i:i+h,j:j+w]*K).sum()
    return Y

X=torch.ones((6,8))
X[:,2:6]=0
print(X)
K=torch.tensor([[1.0,-1.0]])
Y=corr2d(X,K)
print(Y)

conv2d=nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
X=X.reshape((1,1,6,8))
Y=Y.reshape((1,1,6,7))

for i in range(100):
    Y_hat=conv2d(X)
    l=(Y_hat-Y)**2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:]-=3e-2*conv2d.weight.grad
    if (i+2)%2==0:
        print(f'batch {i+1},loss {l.sum():.3f}')

print(conv2d.weight.data.reshape((1,2)))

def comp_conv2d(conv2d,X):
    X=X.reshape((1,1)+X.shape)
    Y=conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d=nn.Conv2d(1,1,kernel_size=(3,5),padding=(0,1),stride=(3,4))
X=torch.rand(size=(8,8))
print(comp_conv2d(conv2d,X).shape)

def corr2d_multi_in(X,K):
    return sum(d2l.corr2d(x,k) for x,k in zip(X,K))

def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)

def corr2d_multi_in_out_1x1(X,K):
    c_i,h,w=X.shape
    c_o=K.shape[0]
    X=X.reshape((c_i,h*w))
    K=K.reshape((c_o,c_i))
    Y=torch.matmul(K,X)
    return Y.reshape((c_o,h,w))

X=torch.tensor([[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]],
                [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]])
K=torch.tensor([[[0.0,1.0],[2.0,3.0]],[[1.0,2.0],[3.0,4.0]]])
K=torch.stack((K,K+1,K+2),0)
print(corr2d_multi_in_out(X,K))

X=torch.normal(0,1,(3,3,3))
K=torch.normal(0,1,(2,3,1,1))
Y1=corr2d_multi_in_out_1x1(X,K)
Y2=corr2d_multi_in_out(X,K)
print(float(torch.abs(Y1-Y2).sum())<1e-6)

import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X,pool_size,mode='max'):
    p_h,p_w=pool_size
    Y=torch.zeros((X.shape[0]-p_h+1,X.shape[1]-p_w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode=='max':
                Y[i,j]=X[i:i+p_h,j:j+p_w].max()
            elif mode=='avg':
                Y[i,j]=X[i:i+p_h,j:j+p_w].mean()
    return Y

X=torch.arange(16,dtype=torch.float32).reshape((1,1,4,4))
X=torch.cat((X,X+1),1)
pool2d=nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d(X))
