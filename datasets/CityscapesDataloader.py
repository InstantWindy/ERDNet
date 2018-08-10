import numpy as np
import os
import torch

from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

colors = [
              [128,  64, 128],
              [244,  35, 232],
              [ 70,  70,  70],
              [102, 102, 156],
              [190, 153, 153],
              [153, 153, 153],
              [250, 170,  30],
              [220, 220,   0],
              [107, 142,  35],
              [152, 251, 152],
              [  0, 130, 180],
              [220,  20,  60],
              [255,   0,   0],
              [  0,   0, 142],
              [  0,   0,  70],
              [  0,  60, 100],
              [  0,  80, 100],
              [  0,   0, 230],
              [119,  11,  32],
             [0,  0, 0 ]]

label_colours = dict(zip(range(20), colors))

def decode_segmap( temp,plot=False):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0,20):
            r[temp == l] = label_colours[l][0]
            g[temp == l] = label_colours[l][1]
            b[temp == l] = label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb




def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_gtFine_labelIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)




class cityscapes(Dataset):

    def __init__(self, root='/home/lulu/dataset/Cityscapes', co_transform=None, input_transform=None,target_transform=None,subset='train'):
        self.images_root = os.path.join(root, 'gtFine/images/')
        self.labels_root = os.path.join(root, 'gtFine/labels/')
        
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',\
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',\
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
        
        self.ignore_index =255
        self.class_map = dict(zip(self.valid_classes, range(19))) 
        
        self.images_root += subset
        self.labels_root += subset

        print (self.images_root)
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS
        self.input_transform=input_transform
        self.target_transform=target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('L')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)
            
        if self.input_transform is not None:
            image=self.input_transform(image)
            
        if self.target_transform is not None:
            label=self.target_transform(label)
            
        label=self.encode_segmap(label.squeeze().numpy())
        
        label[label==255]=19
        label=torch.from_numpy(label)
        return image, label

    def __len__(self):
        return len(self.filenames)
    
    def encode_segmap(self, mask):
        #Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask==_voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask==_validc] = self.class_map[_validc]
        return mask


class TestDataSet(Dataset):

    def __init__(self, root='/home/lulu/dataset/Cityscapes',input_transform=None, subset='test'):
        self.images_root = os.path.join(root, 'gtFine/images/')
        self.images_root += subset
        print (self.images_root)
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()
        self.input_transform=input_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
            
        if self.input_transform is not None:
            image = self.input_transform(image)
        #image = ToTensor()(image)
        return image

    def __len__(self):
        return len(self.filenames)
    
 