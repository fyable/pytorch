import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.utils
import torch.utils.data
import torchvision
from d2l import torch as d2l
from VOCdataset import VOCSegDataset

d2l.use_svg_display()

def load_data_voc(batch_size,crop_size):
    voc_dir=d2l.download_extract("voc2012",os.path.join('VOCdevkit','VOC2012'))
    voc_train = VOCSegDataset(True, crop_size, voc_dir)
    voc_test = VOCSegDataset(False, crop_size, voc_dir)
    train_iter=torch.utils.data.DataLoader(voc_train,batch_size,shuffle=True,drop_last=True,num_workers=4)
    test_iter=torch.utils.data.DataLoader(voc_test,batch_size,shuffle=True,drop_last=True,num_workers=4)
    return train_iter,test_iter

batch_size = 64
crop_size=(320, 480)

train_iter,test_iter=load_data_voc(batch_size,crop_size)
for X,Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break

# n=5
# imgs=[]
# for _ in range(n):
#     imgs+=voc_rand_crop(train_features[0],train_labels[0],200,300)
# imgs = [img.permute(1,2,0) for img in imgs]
# d2l.show_images(imgs[::2]+imgs[1::2], 2, n)
# d2l.plt.show()


# y=voc_label_indices(train_labels[0],voc_colormap2label())
# print(y[105:115, 130:140], VOC_CLASSES[1])

# n=5
# imgs = train_features[0:n] + train_labels[0:n]
# imgs = [img.permute(1,2,0) for img in imgs]
# d2l.show_images(imgs, 2, n)
# d2l.plt.show()