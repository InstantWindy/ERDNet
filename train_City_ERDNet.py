import os
import sys
sys.path.append('../..')
os.environ['TF_CPP_MIN_LOGLEVEL']="2"
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="3"

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import numpy as np
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from mxtorch.vision.eval_tools import eval_semantic_segmentation
from torchvision import models
from datasets.CityscapesDataloader import *
from models.ERDNet import *
from utils import *

from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage


from PIL import Image,ImageOps
import matplotlib.pyplot as plt

#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self,  augment=True, height=512):
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)
       
        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target

class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)
NUM_CLASSES=20
datadir='/home/lulu/dataset/Cityscapes'
co_transform = MyCoTransform(augment=True, height=512)#1024)
co_transform_val = MyCoTransform( augment=False, height=512)#1024)
dataset_train = cityscapes(datadir, co_transform, 'train')
dataset_val = cityscapes(datadir, co_transform_val, 'val')

loader = DataLoader(dataset_train, num_workers=2, batch_size=4, shuffle=True)
loader_val = DataLoader(dataset_val, num_workers=2, batch_size=4, shuffle=False)

weight = torch.ones(NUM_CLASSES)

weight[0] = 2.8149201869965
weight[1] = 6.9850029945374
weight[2] = 3.7890393733978
weight[3] = 9.9428062438965
weight[4] = 9.7702074050903
weight[5] = 9.5110931396484
weight[6] = 10.311357498169
weight[7] = 10.026463508606
weight[8] = 4.6323022842407
weight[9] = 9.5608062744141
weight[10] = 7.8698215484619
weight[11] = 9.5168733596802
weight[12] = 10.373730659485
weight[13] = 6.6616044044495
weight[14] = 10.260489463806
weight[15] = 10.287888526917
weight[16] = 10.289801597595
weight[17] = 10.405355453491
weight[18] = 10.138095855713

weight[19] = 0

weight = weight.cuda()
criterion = CrossEntropyLoss2d(weight)

model_path='Checkpoints_City_ERDNet'
LR=5e-4
epochs=800

net=RDN(num_classes=NUM_CLASSES,layer_block=[3,4,6]).cuda()
#net.load_state_dict(torch.load('Checkpoints_City_BN/16_best-model.pkl'))


# 定义 loss 和 optimizer
#optimizer=torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9,weight_decay=1e-4)
#lr_scheduler=PolyDecay(LR,0.9,epochs)
optimizer = torch.optim.Adam(net.parameters(), LR, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4) 
lambda1 = lambda epoch: pow((1-((epoch-1)/epochs)),0.9)  ## scheduler 2
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) 


automated_log_path='logs/'+'ERDNet/'+'automated_log_city.txt'
if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
       with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tVal-loss\t\tTrain-IoU\t\tVal-IoU\t\tlearningRate")


best_iou=0.
train_losses=[]
val_losses=[]

for e in range(epochs):
    scheduler.step(e)
    #lr_scheduler.scheduler(e,optimizer)
    train_loss = 0
    train_acc = 0
    train_mean_iu = 0
    iouTrain=0.
    
    prev_time = datetime.now()
    net = net.train()
    iouEvalTrain = iouEval(NUM_CLASSES)
    for data in loader :
        im = Variable(data[0].cuda())
        labels = Variable(data[1].cuda())
        # forward
        out = net(im)
        loss = criterion(out, labels)
        train_loss += loss.data[0]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        iouEvalTrain.addBatch(out.max(1)[1].unsqueeze(1).data, labels.unsqueeze(1).data)
#         pred_labels = out.max(dim=1)[1].data.cpu().numpy()
#         pred_labels = [i for i in pred_labels]

#         true_labels = labels.data.cpu().numpy()
#         true_labels = [i for i in true_labels]

#         eval_metrics = eval_semantic_segmentation(pred_labels, true_labels)
        
#         train_acc += eval_metrics['mean_class_accuracy']
#         train_mean_iu += eval_metrics['miou']
        
        
        
#     iouTrain=train_mean_iu/len(loader)
#     iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
#     print ("EPOCH IoU on TRAIN set: ", iouStr, "%") 
#     print('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}'.format(e,train_loss / len(loader),train_acc / len(loader)))
#     print()
    
        
    iouTrain, iou_classes = iouEvalTrain.getIoU()
    iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
    print ("EPOCH IoU on TRAIN set: ", iouStr, "%")    
    print('Epoch: {}, Train Loss: {:.5f}'.format(e,train_loss / len(loader)))
    print()
          
    net = net.eval()
    eval_loss = 0
    eval_acc = 0
    eval_mean_iu = 0
    iouVal=0.
    
    iouEvalVal = iouEval(NUM_CLASSES)
    
    for data in loader_val: 
        im = Variable(data[0].cuda(), volatile=True)
        labels = Variable(data[1].cuda(), volatile=True)
   
        # forward
        out = net(im)
        loss = criterion(out, labels)
        eval_loss += loss.data[0]
        
        iouEvalVal.addBatch(out.max(1)[1].unsqueeze(1).data, labels.unsqueeze(1).data)
#         pred_labels = out.max(dim=1)[1].data.cpu().numpy()
#         pred_labels = [i for i in pred_labels]

#         true_labels = labels.data.cpu().numpy()
#         true_labels = [i for i in true_labels]

#         eval_metrics = eval_semantic_segmentation(pred_labels, true_labels)

        
#     eval_acc += eval_metrics['mean_class_accuracy']
#     eval_mean_iu += eval_metrics['miou']
#     iouVal=eval_mean_iu
    
#     iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
#     print ("EPOCH IoU on VAL set: ", iouStr,"%" ) 
    
   
    iouVal, iou_classes = iouEvalVal.getIoU()
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("EPOCH IoU on VAL set: ", iouStr,"%" ) 
   
    
    if best_iou<iouVal:
        best_iou=iouVal
        with open('logs/' +'ERDNet'+ "/best_val_iou_city.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (e, iouVal))
        print("{}_best-model.pkl".format(os.path.join(model_path,str(e))))
        torch.save(net.state_dict(),"{}_best-model.pkl".format(os.path.join(model_path,str(e))))
        
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)

    epoch_str = ('Epoch: {}, Valid Loss: {:.5f}, Valid Acc: {:.5f}'.format(e, eval_loss / len(loader_val),eval_acc))
    
    time_str = ', Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    train_losses.append(train_loss / len(loader))
    val_losses.append(eval_loss / len(loader_val))
    print(epoch_str + time_str + ' , lr: {}'.format(optimizer.param_groups[0]['lr']))
    print()
    usedLr=float(optimizer.param_groups[0]['lr'])
    with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.8f" % (e, train_loss / len(loader), eval_loss / len(loader_val), iouTrain, iouVal, usedLr ))
