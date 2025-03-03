import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l

torch.set_printoptions(2)

d2l.DATA_HUB['banana-detection']=(d2l.DATA_URL+'banana-detection.zip','5de26c8fce5ccdea9f91267273464dc968d20d72')

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

#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

def multibox_prior(data,sizes=[0.75,0.5,0.25],ratios=[1,2,0.5]):
    in_height,in_width=data.shape[-2:]
    device,num_sizes,num_ratios=data.device,len(sizes),len(ratios)
    boxes_per_pixel=(num_sizes+num_ratios-1)
    size_tensor=torch.tensor(sizes,device=device)
    ratio_tensor=torch.tensor(ratios,device=device)
    offset_h,offset_w=0.5,0.5
    steps_h=1.0/in_height
    steps_w=1.0/in_width
    center_h=(torch.arange(in_height,device=device)+offset_h)*steps_h
    center_w=(torch.arange(in_width,device=device)+offset_w)*steps_w
    shift_y,shift_x=torch.meshgrid(center_h,center_w,indexing='ij')
    shift_y,shift_x=shift_y.reshape(-1),shift_x.reshape(-1)
    
    w=torch.cat((size_tensor*torch.sqrt(ratio_tensor[0]),sizes[0]*torch.sqrt(ratio_tensor[1:])))
    h=torch.cat((size_tensor/torch.sqrt(ratio_tensor[0]),sizes[0]/torch.sqrt(ratio_tensor[1:])))
    anchor_manipulations=torch.stack((-w,-h,w,h)).T.repeat(in_height*in_width,1)/2
    out_grid=torch.stack((shift_x,shift_y,shift_x,shift_y),dim=1).repeat_interleave(boxes_per_pixel,dim=0)
    output=out_grid+anchor_manipulations
    return output.unsqueeze(0)
    
def box_iou(boxes1,boxes2):
    box_area=lambda boxes:(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    area1,area2=box_area(boxes1),box_area(boxes2)
    inter_lefts=torch.max(boxes1[:,None,:2],boxes2[:,:2])
    inter_rights=torch.min(boxes1[:,None,2:],boxes2[:,2:])
    inters=(inter_rights-inter_lefts).clamp(min=0)
    inter_areas=inters[:,:,0]*inters[:,:,1]
    union=area1[:,None]+area2-inter_areas
    return inter_areas/union

def assign_anchor_to_bbox(ground_truth,anchors,device,iou_threshold=0.5):
    num_anchors,num_gt_boxes=anchors.shape[0],ground_truth.shape[0]
    jaccard=box_iou(anchors,ground_truth)
    anchors_bbox_map=torch.full((num_anchors,),-1,dtype=torch.long,device=device)
    
    max_ious,indices=torch.max(jaccard,dim=1)
    anc_i=torch.nonzero(max_ious>=iou_threshold).reshape(-1)
    box_j=indices[max_ious>=iou_threshold]
    anchors_bbox_map[anc_i]=box_j
    col_discard=torch.full((num_anchors,),-1)
    row_discard=torch.full((num_gt_boxes,),-1)
    
    for _ in range(num_gt_boxes):
        max_idx=torch.argmax(jaccard)
        box_idx=(max_idx%num_gt_boxes).long()
        anc_idx=(max_idx//num_gt_boxes).long()
        anchors_bbox_map[anc_idx]=box_idx
        # print(anc_idx,box_idx)
        jaccard[:,box_idx]=col_discard
        jaccard[anc_idx,:]=row_discard
    # print(1)
    # print(anchors_bbox_map,111) 
    return anchors_bbox_map

def offset_boxes(anchors,assigned_bb,eps=1e-6):
    c_anc=d2l.box_corner_to_center(anchors)
    c_assigned_bb=d2l.box_corner_to_center(assigned_bb)
    offset_xy=10*(c_assigned_bb[:,:2]-c_anc[:,:2])/c_anc[:,2:]
    offset_wh=5*torch.log(eps+c_assigned_bb[:,2:]/c_anc[:,2:])
    offset=torch.cat((offset_xy,offset_wh),1)
    return offset

#@save
def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        # print(indices_true)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

def offset_inverse(anchors,offset_preds):
    anc=d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)

img=d2l.plt.imread('../img/cat.jpg')
h,w=img.shape[:2]
print(h,w)
X=torch.rand(size=(1,3,h,w))
Y=multibox_prior(X,sizes=[0.75,0.5,0.25],ratios=[1,2,0.5])
boxes=Y.reshape(h,w,5,4)
d2l.set_figsize()
bbox_scale=torch.tensor((w,h,w,h))
# fig=d2l.plt.imshow(img)
# show_bboxes(fig.axes,boxes[800,1000,:,:]*bbox_scale,['s=0.75,r=1','s=0.5,r=1','s=0.25,r=1','s=0.75,r=2','s=0.75,r=0.5'])
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
# d2l.plt.show()

labels = multibox_target(anchors.unsqueeze(dim=0),ground_truth.unsqueeze(dim=0))
print(labels[0])