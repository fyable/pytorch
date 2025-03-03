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
            l=loss(y_hat, y).sum()
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

def bilinear_kernel(in_channels,out_channels,kernel_size):
    factor=(kernel_size+1)//2
    if kernel_size%2==1:
        center=kernel_size//2
    else:
        center=kernel_size//2-0.5
    og=(torch.arange(kernel_size).reshape(-1,1),torch.arange(kernel_size).reshape(1,-1))
    filt=(1-torch.abs(og[0]-center)/factor)*(1-torch.abs(og[1]-center)/factor)
    weight=torch.zeros((in_channels,out_channels,kernel_size,kernel_size))
    weight[range(in_channels),range(out_channels),:,:]=filt
    return weight

pretrain_net=torchvision.models.resnet18(pretrained=True)
# print(list(pretrain_net.children())[-3:])
net=nn.Sequential(*list(pretrain_net.children())[:-2])
X=torch.rand(size=(1,3,320,480))
num_classes=21
net.add_module('final_conv',nn.Conv2d(512,num_classes,kernel_size=1))
net.add_module('transpose_conv',nn.ConvTranspose2d(num_classes,num_classes,kernel_size=64,padding=16,stride=32))
W=bilinear_kernel(num_classes,num_classes,64)
net.transpose_conv.weight.data.copy_(W)

lr=1e-3
batch_size,num_epochs,crop_size=32,5,(320,480)

devices=d2l.try_all_gpus()
    
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets,reduction='none').mean(1).mean(1)

# print(loss(net(X),torch.zeros((1,320,480),dtype=torch.long)).shape)
# params_1x=[param for name,param in net.named_parameters() if name not in ['transpose_conv.weight','transpose_conv.bias']]
optimizer=torch.optim.SGD(net.parameters(),lr=lr,weight_decay=0.001)

train_iter,test_iter=d2l.load_data_voc(batch_size,crop_size)
train(net,train_iter,test_iter,num_epochs,devices,optimizer,loss)

def predict(img):
    plt.ioff()
    X=test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred=net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape((pred.shape[1],pred.shape[2]))

def label2image(pred):
    colormap=torch.tensor(d2l.VOC_COLORMAP,device=devices[0])
    X=pred.long()
    return colormap[X,:]

voc_dir=d2l.download_extract('voc2012','VOCdevkit/VOC2012')
test_images,test_labels=d2l.read_voc_images(voc_dir,False)
n,imgs=4,[]
for i in range(n):
    crop_rect=(0,0,320,480)
    X=torchvision.transforms.functional.crop(test_images[i],*crop_rect)
    pred=label2image(predict(X))
    imgs+=[X.permute(1,2,0),pred.cpu(),torchvision.transforms.functional.crop(test_labels[i],*crop_rect).permute(1,2,0)]

d2l.show_images(imgs[::3]+imgs[1::3]+imgs[2::3],3,n,scale=2)
d2l.plt.show()


# conv_trans=nn.ConvTranspose2d(3,3,kernel_size=4,padding=1,stride=2,bias=False)
# conv_trans.weight.data=bilinear_kernel(3,3,4)
# img=torchvision.transforms.ToTensor()(d2l.Image.open('img/cat.jpg'))
# X=img.unsqueeze(0)
# Y=conv_trans(X)
# out_img=Y[0].permute(1,2,0).detach()
# d2l.set_figsize()

# print(img.permute(1,2,0).shape)
# print(out_img.shape)

# d2l.plt.imshow(out_img)
# d2l.plt.show()