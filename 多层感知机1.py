import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

batch_size=256
num_workers=4

num_input=784
num_outputs=10
num_hiddens=256

lr=0.1
num_epochs=30

def load_data_fashion_mnist(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train=torchvision.datasets.FashionMNIST(root='../data',train=True,transform=trans,download=True)
    mnist_test=torchvision.datasets.FashionMNIST(root='../data',train=False,transform=trans,download=True)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=num_workers),
            data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=num_workers))

def get_fashion_mnist_labels(labels):
    text_labels=['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    figsize=(num_cols*scale,num_rows*scale)
    _,axes=d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    axes=axes.flatten()
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition

def relu(X):
    a=torch.zeros_like(X)
    return torch.max(X,a)

def sigmod(X):
    a=torch.ones_like(X)
    return a/(a+(-X).exp())

def net(X):
    X=X.reshape((-1,num_input))
    H=sigmod(X@W1+b1)
    return (H@W2+b2)

def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)
    cmp=(y_hat.type(y.dtype)==y)
    return float(cmp.type(y.dtype).sum())

class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b)for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric=Accumulator(2)
    for X,y in data_iter:
        metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric=Accumulator(3)
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l)*len(y),accuracy(y_hat,y),y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()),accuracy(y_hat, y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    for epoch in range(num_epochs):
        train_metrics=train_epoch_ch3(net,train_iter,loss,updater)
        test_acc=evaluate_accuracy(net,test_iter)
        print(test_acc)

def predict_ch3(net,test_iter,n=50):
    for X,y in test_iter:
        break
    trues=d2l.get_fashion_mnist_labels(y)
    preds=d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles=[true+'\n'+pred for true,pred in zip(trues,preds)]
    d2l.show_images(X[0:n].reshape((n,28,28)),5,n//5,titles=titles[0:n])
    d2l.plt.show()

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

W1=nn.Parameter(torch.randn(num_input,num_hiddens,requires_grad=True))
b1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))

W2=nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True))
b2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

params=[W1,b1,W2,b2]

loss=nn.CrossEntropyLoss()
trainer=torch.optim.SGD(params,lr=lr)


train_iter,test_iter=load_data_fashion_mnist(batch_size,resize=None)
train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)
predict_ch3(net,test_iter)
