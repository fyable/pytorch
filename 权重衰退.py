import torch
import torchvision
from torch import nn
from torch.nn.functional import dropout
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

lr = 0.003
num_epochs = 50
wd= 50

n_train,n_test,num_input,batch_size=20,100,200,5
true_w,true_b=torch.ones((num_input,1))*0.01,0.05

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    net.eval()
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(nn.MSELoss()(net(X), y), y.numel())
    return metric[0]/metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    net.train()
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.backward()
        updater.step()

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    test_acc=0
    for epoch in range(num_epochs):
        train_epoch_ch3(net, train_iter, loss, updater)

        train_acc = evaluate_accuracy(net, train_iter)
        print("train:")
        print(train_acc)

        test_acc = evaluate_accuracy(net, test_iter)
        print("test:")
        print(test_acc)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

train_data=d2l.synthetic_data(true_w,true_b,n_train)
train_iter=d2l.load_array(train_data,batch_size)

test_data=d2l.synthetic_data(true_w,true_b,n_test)
test_iter=d2l.load_array(test_data,batch_size,is_train=False)

net = nn.Sequential(nn.Flatten(),nn.Linear(num_input, 1))
net.apply(init_weights)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
train_ch3(net, train_iter, test_iter, loss, num_epochs,  trainer)


