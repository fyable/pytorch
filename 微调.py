import matplotlib.pyplot as plt
import os
import torch
import torch.utils
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from IPython import display

plt.ion()
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
        display.display(self.fig)
        plt.pause(0.1)
        display.clear_output(wait=True)

def get_labels(labels):
    text_labels=['hotdog','not_hotdog']
    return [text_labels[int(i)] for i in labels]

def predict(net,test_iter,n=10):
    plt.ioff()
    net.eval()
    for X,y in test_iter:
        break
    X=X.to(device=next(iter(net.parameters())).device)
    trues=get_labels(y)
    preds=get_labels(net(X).argmax(axis=1))
    titles=[true+'\n'+pred for true,pred in zip(trues,preds)]
    X=X.permute(0,2,3,1)
    X=X.cpu()
    d2l.show_images(X[0:n],1,n,titles=titles[0:n])
    d2l.plt.show()

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

def train(net,train_iter,test_iter,num_epochs,devices=d2l.try_all_gpus(),optimizer=None,loss=nn.CrossEntropyLoss()):
    net=nn.DataParallel(net,device_ids=devices).to(devices[0])
    animator=Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.0, 1.0],legend=['train loss','train acc','test acc'])
    timer,num_batches=d2l.Timer(),len(train_iter)
    for epoch in range(num_epochs):
        metric=d2l.Accumulator(3)
        net.train()
        for i,(X,y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X,y=X.to(devices[0]),y.to(devices[0])
            y_hat=net(X)
            l=loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l, d2l.accuracy(y_hat, y), y.numel())
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices[0])}')

normalize=torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

train_augs=torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),normalize])

test_augs=torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),normalize])

def load_data(is_train,path,augs,batch_size):
    dataset=torchvision.datasets.ImageFolder(path,transform=augs)
    return torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=is_train,num_workers=4)

def train_with_data_aug(train_augs,test_augs,lr):
    net=torchvision.models.resnet18(pretrained=True)
    net.fc=nn.Linear(net.fc.in_features,2)
    nn.init.xavier_uniform_(net.fc.weight)
    
    batch_size,num_epochs=256,5
    
    train_iter=load_data(True,os.path.join(data_dir,'train'),train_augs,batch_size)
    test_iter=load_data(False,os.path.join(data_dir,'test'),test_augs,batch_size)
    
    devices=d2l.try_all_gpus()
    
    params_1x=[param for name,param in net.named_parameters() if name not in ['fc.weight','fc.bias']]
    optimizer=torch.optim.SGD([{'params':params_1x},
                               {'params':net.fc.parameters(),'lr':lr*10}],
                              lr=lr,weight_decay=0.001)
    
    loss=nn.CrossEntropyLoss(reduction="sum")
    
    train(net,train_iter,test_iter,num_epochs,devices,optimizer,loss)
    # predict(net,test_iter)

d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip','fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir=d2l.download_extract('hotdog')
# train_imgs=torchvision.datasets.ImageFolder(os.path.join(data_dir,'train'))
# test_imgs=torchvision.datasets.ImageFolder(os.path.join(data_dir,'test'))

lr=5e-5
train_with_data_aug(train_augs,test_augs,lr)