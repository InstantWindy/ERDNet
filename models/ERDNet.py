import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable

class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)
    def forward(self, x):
        x = self.body(x)
        return x
        
class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3,dilation=1):
        super(make_dense, self).__init__()
        ##3 x 3 卷积分成两个1x3 ,3 x 1卷积
        self.conv3x1_1 = nn.Conv2d(nChannels, growthRate, (kernel_size, 1), stride=1, padding=(1*dilation,0), bias=False, dilation = (dilation,1))
        self.conv1x3_1 = nn.Conv2d(growthRate, growthRate, (1,kernel_size), stride=1, padding=(0,1*dilation), bias=False, dilation = (1, dilation))
        self.bn = nn.BatchNorm2d(growthRate, eps=1e-03)
    def forward(self, x):
        out = self.conv3x1_1(x)
        out = F.relu(out)
        out = self.conv1x3_1(out)
        out = self.bn(out)
        out = F.relu(out)

        out = torch.cat((x, out), 1)
        return out

class _Transition(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(_Transition, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=2,stride=2,bias=False)
        )
    def forward(self,x):
        return self.conv(x)
    
# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate,stride=1,dilation=1):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate,kernel_size=3,dilation=dilation))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)    
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, stride=1,padding=0, bias=False)
        self.dropout = nn.Dropout2d(0.1)
    
        self.fc1 = nn.Linear(in_features= nChannels, out_features=round(nChannels / 4))
        self.fc2 = nn.Linear(in_features=round(nChannels / 4), out_features=nChannels )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        if (self.dropout.p != 0):     #添加dropout
            out = self.dropout(out)
        
        original_out = out  
        out = F.avg_pool2d(input=out,kernel_size=(out.size(2),out.size(3)),stride=1) 
        out = out.view(out.size(0), -1)
        out = self.fc1(out)   
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0),out.size(1),1,1)
        out = out * original_out
        
        out = out + x
        return out

# Residual Dense Network
class RDN(nn.Module):
    def __init__(self,num_classes=20,layer_block=[3,4,6]):
        super(RDN, self).__init__()
        nChannel = 3 
        nFeat =128 
        scale = 2 
        growthRate =32 
        num_classes=num_classes
        layer_block=layer_block
        # F-1
        self.conv1 = nn.Conv2d(nChannel,64, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(64, nFeat, kernel_size=3, padding=1,stride=2, bias=True)
        # RDBs 3 
        self.RDB1 = RDB(nFeat, layer_block[0],growthRate)
        self.trans1=_Transition(nFeat,nFeat)  #1/2
        self.RDB2 = RDB(nFeat,layer_block[1], growthRate,dilation=2)
        self.trans2=_Transition(nFeat,nFeat)  #1/2
        self.RDB3 = RDB(nFeat, layer_block[2], growthRate,dilation=4)

        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*2, nFeat, kernel_size=1, stride=1,padding=0, bias=True)
        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat*scale*scale, kernel_size=3,stride=1, padding=1, bias=True)
        self.upsample = sub_pixel(scale)
        # conv 
        self.conv3 = nn.Conv2d(nFeat, num_classes, kernel_size=1, padding=0, bias=True)
    def forward(self, x):

        F_  = self.conv1(x)
        F_0 = self.conv2(F_)     #1/2
        F_1 = self.RDB1(F_0)
        F_1_1 = self.trans1(F_1) #1/4
        F_2 = self.RDB2(F_1_1)
        F_2_2= self.trans2(F_2) #1/8
        F_3 = self.RDB3(F_2_2)   

        
        F_3=self.conv_up(F_3)
        F_3=self.upsample(F_3)
        FF_1=torch.cat((F_2,F_3),1) #1/4
        FGF_1 = self.GFF_1x1(FF_1)         

        
        FGF_1=self.conv_up(FGF_1)
        FGF_1=self.upsample(FGF_1) #1/2
        FF_2=torch.cat((F_1,FGF_1),1) #1/2
        FGF_2 = self.GFF_1x1(FF_2)         

        
        FGF_2=self.conv_up(FGF_2)
        FGF_2=self.upsample(FGF_2) #1
        output = self.conv3( FGF_2)


        return output

