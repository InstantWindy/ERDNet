import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils import data
# from ptsemseg.augmentations import *

class CamvidLoader(data.Dataset):
    def __init__(self, root, split="train", 
                 is_transform=False, img_size=None, augmentations=None, img_norm=True):
        self.root = root
        self.split = split
        self.img_size = [360, 480]
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.n_classes = 12
        self.files = collections.defaultdict(list)

    
        file_list = os.listdir(root + '/' +'images'+'/'+ self.split)
        self.files[self.split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root +'/'+'images'+ '/' + self.split + '/' + img_name
        lbl_path = self.root + '/' +'label'+'/'+ self.split +'/'+ img_name


        img = Image.open(img_path)
        lbl = Image.open(lbl_path)

        
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)  # Image

        if self.is_transform:
            img = np.array(img, dtype=np.uint8)
            lbl = np.array(lbl, dtype=np.int64)
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

def decode_segmap( temp, plot=False):
        Sky = [128, 128, 128]
        Building = [128, 0, 0]
        Pole = [192, 192, 128]
        Road_marking = [255, 69, 0]
        Road = [128, 64, 128]
        Pavement = [60, 40, 222]
        Tree = [128, 128, 0]
        SignSymbol = [192, 128, 128]
        Fence = [64, 64, 128]
        Car = [64, 0, 128]
        Pedestrian = [64, 64, 0]
        Bicyclist = [0, 128, 192]
        Unlabelled = [0, 0, 0]

        label_colours = np.array([Sky, Building, Pole, Road, 
                                  Pavement, Tree, SignSymbol, Fence, Car, 
                                  Pedestrian, Bicyclist, Unlabelled])
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, 12):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb