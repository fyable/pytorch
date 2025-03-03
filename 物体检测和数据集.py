import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l

def box_corner_to_center(boxes):
    x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    cx=(x1+x2)/2
    cy=(y1+y2)/2
    w=x2-x1
    h=y2-y1
    boxes=torch.stack((cx,cy,w,h),1)
    return boxes

def box_center_to_corner(boxes):
    cx,cy,w,h=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    x1=cx-w/2
    y1=cy-h/2
    x2=cx+w/2
    y2=cy+h/2
    boxes=torch.stack((x1,y1,x2,y2),1)
    return boxes

def bbox_to_rect(bbox,color):
    return d2l.plt.Rectangle(xy=(bbox[0],bbox[1]),width=bbox[2]-bbox[0],height=bbox[3]-bbox[1],fill=False,edgecolor=color,linewidth=2)

def read_data_bananas(is_train=True):
    data_dir=d2l.download_extract('banana-detection')
    csv_fname=os.path.join(data_dir,'bananas_train' if is_train else 'bananas_val','label.csv')
    csv_data=pd.read_csv(csv_fname)
    csv_data=csv_data.set_index('img_name')
    images,targets=[],[]
    for img_name,target in csv_data.iterrows():
        images.append(torchvision.io.read_image(os.path.join(data_dir,'bananas_train' if is_train else 'bananas_val','images',f'{img_name}')))
        targets.append(list(target))
    return images,torch.tensor(targets).unsqueeze(1)/256

class BananasDataset(torch.utils.data.Dataset):
    def __init__(self,is_train):
        self.features,self.labels=read_data_bananas(is_train)
        print('read '+str(len(self.features))+(f' train examples' if is_train else ' validation examples'))
    def __getitem__(self,idx):
        return (self.features[idx].float(),self.labels[idx])
    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    train_iter=torch.utils.data.DataLoader(BananasDataset(is_train=True),batch_size,shuffle=True)
    val_iter=torch.utils.data.DataLoader(BananasDataset(is_train=False),batch_size)
    return train_iter,val_iter

d2l.DATA_HUB['banana-detection']=(d2l.DATA_URL+'banana-detection.zip','5de26c8fce5ccdea9f91267273464dc968d20d72')
batch_size,edge_size=32,256
train_iter,val_iter=load_data_bananas(batch_size)
batch=next(iter(train_iter))
print(batch[0].shape,batch[1].shape)
imgs=(batch[0][0:10].permute(0,2,3,1))/255
axes=d2l.show_images(imgs,2,5,scale=2)
for ax,label in zip(axes,batch[1][0:10]):
    d2l.show_bboxes(ax,[label[0][1:5]*edge_size],colors=['w'])
d2l.plt.show()

# d2l.set_figsize()
# img=plt.imread('../img/cat.jpg')
# d2l.plt.imshow(img)
# d2l.plt.show()
# dog_bbox,cat_bbox=[60,45,378,516],[400,112,655,493]
# fig=d2l.plt.imshow(img)
# fig.axes.add_patch(bbox_to_rect(dog_bbox,'blue'))
# fig.axes.add_patch(bbox_to_rect(cat_bbox,'red'))
# d2l.plt.show()
# boxes=torch.tensor((dog_bbox,cat_bbox))
# print(box_center_to_corner(box_corner_to_center(boxes))==boxes)