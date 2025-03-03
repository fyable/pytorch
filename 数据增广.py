import matplotlib.pyplot as plt
import torch
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
    text_labels=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    return [text_labels[int(i)] for i in labels]

def predict(net,test_iter,n=10):
    plt.ioff()
    net.eval()
    for X,y in test_iter:
        break
    print(X.shape)
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

train_augs=torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()])

test_augs=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])

def load_cifar10(is_train,augs,batch_size):
    dataset=torchvision.datasets.CIFAR10(root="../data",train=is_train,transform=augs,download=True)
    return torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=is_train,num_workers=4)

def train_with_data_aug(train_augs,test_augs,lr=0.001):
    net=d2l.resnet18(10,3)
    net.apply(init_weights)
    batch_size,num_epochs=256,10
    train_iter=load_cifar10(True,train_augs,batch_size)
    test_iter=load_cifar10(False,test_augs,batch_size)
    devices=d2l.try_all_gpus()
    optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    loss=nn.CrossEntropyLoss(reduction="sum")
    train(net,train_iter,test_iter,num_epochs,devices,optimizer,loss)
    predict(net,test_iter)

lr=0.001
train_with_data_aug(train_augs,test_augs,lr)

# d2l.set_figsize()
# img=d2l.Image.open('../pytorch/img/cat.jpg')
# d2l.plt.imshow(img)
# d2l.plt.show()

# all_images=torchvision.datasets.CIFAR10(train=True,root="../data",download=True)

# def apply(img,aug,num_rows=2,num_cols=4,scale=1.5):
#     Y=[aug(img) for _ in range(num_rows*num_cols)]
#     d2l.show_images(Y,num_rows,num_cols,scale=scale)
#     d2l.plt.show()

# apply(img,aug=torchvision.transforms.RandomHorizontalFlip())
# apply(img,aug=torchvision.transforms.RandomVerticalFlip())
# apply(img,aug=torchvision.transforms.RandomResizedCrop(200,scale=(0.1,1),ratio=(0.5,2)))
# apply(img,aug=torchvision.transforms.ColorJitter(brightness=0.5,contrast=0,saturation=0,hue=0))
# apply(img,aug=torchvision.transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0.5))
# apply(img,aug=torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5))
# apply(img,aug=torchvision.transforms.Compose([
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.RandomResizedCrop(200,scale=(0.1,1),ratio=(0.5,2)),
#     torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5)]))
