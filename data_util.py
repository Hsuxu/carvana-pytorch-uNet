import os
import torch
import cv2
import math
import torchvision.datasets
import numpy as np
import preprocessing as pre
import torch.utils.data as DATA
from PIL import Image
from torch.utils.data import DataLoader
from skimage.segmentation import find_boundaries

ROOT='../Data/'
TRAIN='train'
MASK='train_masks'
TEST='test'
TEST_MASK='test_mask'
NUM_CLASS=2
CLASS_WEIGHT=[7.893,2.107]

class CarDataSet(DATA.Dataset):
    def __init__(self,root,train,mask=None,transform=True,trainable=True):
        self.trainable=trainable
        self.train=os.path.join(root,train)
        if self.trainable:
            self.mask=os.path.join(root,mask)
        
        self.image_list = os.listdir(self.train)
        self.transform = transform
        
    def __getitem__(self,index):
        if self.trainable:
            img_path = os.path.join(self.train, self.image_list[index])
            mask_name = self.image_list[index].split('.')[0] + '_mask.gif'
            mask_path = os.path.join(self.mask, mask_name)
            img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)
            img/=255.0
            mask = np.array(Image.open(mask_path))
            # msk_bdy=find_boundaries(mask).astype(np.int64)
            mask = mask.astype(np.int64)
            # img=cv2.resize(img,)
            if self.transform:
                if np.random.random() < 0.5:
                    img = np.fliplr(img)
                    mask = np.fliplr(mask)
                # if np.random.random() < 0.5:
                #     alpha=np.random.random()*0.6+0.7
                #     beta=1.0*np.random.randint(-50,50)/255.0
                #     img=pre.contrast_adjust(img,alpha,beta)
            return np.transpose(img,[2,0,1]).copy(), mask.copy()
        else:
            img_path = os.path.join(self.train, self.image_list[index])
            img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)
            img /= 255.0
            return np.transpose(img,[2,0,1]).copy(),self.image_list[index]
    def __len__(self):
        return len(self.image_list)


def augmented_train_valid_split(dataset, test_size = 0.15, shuffle = False, random_seed = 0):
    """ Return a list of splitted indices from a DataSet.
    Indices can be used with DataLoader to build a train and validation set.
    
    Arguments:
        A Dataset
        A test_size, as a float between 0 and 1 (percentage split) or as an int (fixed number split)
        Shuffling True or False
        Random seed
    """
    length = len(dataset)
    indices = list(range(1,length))
    
    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    if type(test_size) is float and test_size<1.0:
        split = int(test_size * length)
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or a float' % str)
    return indices[split:], indices[:split]

# data=CarDataSet(ROOT,TRAIN,MASK)
# print(len(data))
