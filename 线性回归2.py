import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l
def synthetic_data(w,b,num_example):
    X=torch.normal(0,1,(num_example,len(w)))
    y=torch.matmul(X,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))
def load_array(data_arrays,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
def linreg(X,w,b):
    return torch.matmul(X,w)+b

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)
d2l.set_figsize()
d2l.plt.scatter(features[:,(1)].detach().numpy(),labels.detach().numpy(),1)
# d2l.plt.show()

batch_size=10
data_iter=load_array((features,labels),batch_size)
next(iter(data_iter))

net=nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

lr=0.01
num_epochs=10

for epoch in range(num_epochs):
    for X,y in data_iter:
        l=nn.MSELoss()(net(X),y)
        torch.optim.SGD(net.parameters(),lr).zero_grad()
        l.backward()
        torch.optim.SGD(net.parameters(),lr).step()
    train_l=nn.MSELoss()(net(features),labels)
    print(f'epoch{epoch+1},loss{float(train_l):f}')
