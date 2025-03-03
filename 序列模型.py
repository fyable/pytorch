import matplotlib.pyplot as plt
import os
import torch
import torch.utils
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from IPython import display
import torchvision.transforms.functional

d2l.set_figsize()
d2l.use_svg_display()  

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.clear_output(wait=True)
        display.display(self.fig)
        plt.pause(0.1)
        
T = 1000
time = torch.arange(1, T +1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
# d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
# d2l.plt.cla()

tau = 50
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),batch_size, is_train=True)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    
def get_net():
    net = nn.Sequential(nn.Linear(tau, 512), nn.ReLU(), nn.Linear(512, 1))
    net.apply(init_weights)
    return net

loss = nn.MSELoss()

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    animator = Animator(xlabel='epoch', ylabel='loss', xlim=[1, epochs], ylim=[0.00, 0.1])
    for epoch in range(1, epochs + 1):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        if epoch % 100 == 0:
            print(f'epoch {epoch}, loss {float(l):.6f}')
        animator.add(epoch, d2l.evaluate_loss(net, train_iter, loss))
        
net = get_net()
train(net, train_iter, loss, 200, 0.001)

d2l.plt.cla()
onestep_preds = net(features)
d2l.plot([time, time[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()], 'time', 'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()

multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(multistep_preds[i - tau: i].reshape((1, -1)))
    
d2l.plt.cla()    
d2l.plot([time, time[tau:], time[n_train + tau:]], 
         [x.detach().numpy(), onestep_preds.detach().numpy(), multistep_preds[n_train + tau:].detach().numpy()], 
         'time', 'x', legend=['data', '1-step preds', 'multistep preds'], xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()

max_steps = 64

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau: i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plt.cla()    
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps], 
         [features[:, tau + i - 1].detach().numpy() for i in steps], 'time', 'x', legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000], figsize=(6, 3))
d2l.plt.show()