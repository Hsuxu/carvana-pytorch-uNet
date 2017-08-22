import cv2
import numpy as np

"""
This is a image preprocessing library for personal use
Is there any problems please concat:
Hsuxu820@gmail.com
"""

def remove_mean(image):
    """
    remove RGB mean values which from ImageNet
    input:
        image:  RGB image np.ndarray 
                type of elements is np.uint8
    return:
        image:  remove RGB mean and scale to [0,1] 
                type of elements is np.float32
    """
    mean = [0.48462227599918,  0.45624044862054, 0.40588363755159]
    image = image.astype(np.float32)
    image = np.subtract(np.divide(image, 255.0), mean)
    return image


def standardize(image, mean=[0.48462227599918,  0.45624044862054, 0.40588363755159], std=[0.22889466674951, 0.22446679341259, 0.22495548344775]):
    """
    standardize RGB mean and std values which from ImageNet
    input:
        image:  RGB image np.ndarray 
                type of elements is np.uint8
    return:
        image:  standarded image
                type of elements is np.float32
    """
    image = image.astype(np.float32) / 255.0
    image = np.divide(np.subtract(image, mean), std)
    return image
    
def samele_wise_normalization(data):
    """
    normalize each sample to 0-1
    Input:
        sample image
    Output:
        Normalized sample
    x=1.0*(x-np.min(x))/(np.max(x)-np.min(x))
    """
    data.astype(np.float32)
    if np.max(data) == np.min(data):
        return np.ones_like(data, dtype=np.float32) * 1e-6
    else:
        return 1.0 * (data - np.min(data)) / (np.max(data) - np.min(data))


def contrast_adjust(image, alpha=1.3, beta=20):
    """
    adjust constrast through gamma correction
    newimg = image * alpha + beta
    input:
        image: np.uint8 or np.float32
    output:
        image: np.uint8 or np.float
    """
    newimage = image.astype(np.float32) * alpha + beta
    
    if type(image[0,0,0])==np.uint8:
        newimage[newimage < 0] = 0
        newimage[newimage > 255] = 255
        return np.uint8(newimage)
    else:
        newimage[newimage < 0] = 0
        newimage[newimage > 1] = 1.
        return newimage

def random_flip(image, lr, ud):
    """
    random flip image 
    """
    if lr:
        if np.random.random() > 0.5:
            image = cv2.flip(image, flipCode=1)
    if ud:
        if np.random.random() > 0.5:
            image = cv2.flip(image, flipCode=0)
    return image


def image_crop(image, crop=None, random_crop=False):
    """
    if crop is None crop size is generated with a random size range from [0.5*height,height]
    if random_crop == True image croped from a random position
    input:
        image: image np.ndarray [H,W,C]
        crop: [target_height,target_width]
    output:
        croped image with shape[crop[0],crop[1],C]
    """
    hei, wid, _ = image.shape
    if crop is None:
        crop = (np.random.randint(int(hei / 2),  hei),
                np.random.randint(int(wid / 2),  wid))
    th, tw = [int(round(x / 2)) for x in crop]
    if random_crop:
        th, tw = np.random.randint(
            0, hei - crop[0] - 1), np.random.randint(0, wid - crop[1] - 1)
    return image[th:th + crop[0], tw:tw + crop[1]]

def image_pad(image,pad_width=None,axis=0,mode='symmetric'):
    """
    pad an image 
    like np.pad way
    input:
        image: ndarray [rgb]
        
    """
    hei,wid=image.shape[0],image.shape[1]
    
    if pad_width is None:
        th=hei//10
        tw=wid//10
        pad_width=((th,th),(tw,tw),(0,0))
    if axis==0:
        if type(pad_width[0])==tuple:
            pad_width=(pad_width[0],(0,0),(0,0))
        else:
            pad_width=(pad_width,(0,0),(0,0))
    if axis==1:
        if type(pad_width[0])==tuple:
            pad_width=((0,0),pad_width[1],(0,0))
        else:
            pad_width=((0,0),pad_width,(0,0))
    if len(image.shape)==3:
        newimage=np.pad(image,pad_width,mode)
    elif len(image.shape)==2:
        newimage=np.squeeze(np.pad(image[:,:,np.newaxis],pad_width,mode))
    
    return cv2.resize(newimage,(wid,hei),interpolation=cv2.INTER_NEAREST)