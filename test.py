import os 
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
from data_util import *
from model import *
from tensorboard import SummaryWriter
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Carvance')
parser.add_argument('--ckpt', type=str, default='checkpoint.pth.tar',
                    help='resume training')
args = parser.parse_args()
args.cuda =torch.cuda.is_available()

csv_fimename='test_submision.csv'

if not os.path.exists(os.path.join(ROOT,TEST_MASK)):
    os.mkdir(os.path.join(ROOT,TEST_MASK))

def rle(img):
    """
    mask to rle encodings
    """
    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)
    flat_img = np.insert(flat_img, [0, len(flat_img)], [0, 0])

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 1
    ends_ix = np.where(ends)[0] + 1
    lengths = ends_ix - starts_ix

    encoding = ''
    for idx in range(len(starts_ix)):
        encoding += '%d %d ' % (starts_ix[idx], lengths[idx])
    return encoding


def test(model,test_loader,save_path):
    model.eval()
    col1=['img']
    col2=['rle_mask']
    t0=time.time()
    for i,(data,img_name) in enumerate(test_loader):
        t1=time.time()
        if args.cuda:
            data = data.float().cuda()
        # print(data.size())
        data = Variable(data, volatile=True)
        output = model(data)
        _, pred = torch.max(output, 1)
        pred=pred.data.cpu().numpy()    
        for i in range(pred.shape[0]):
            encoding=rle(pred[i])
            col1.append(img_name[i])
            col2.append(encoding)
            
            cv2.imwrite(os.path.join(save_path,
                                    img_name[i].split('.')[0] + '_mask.jpg'),
                        np.uint8(pred[i] * 255))
        print(img_name[i], 'each image: {:.4f}s'.format(time.time() - t1))
    data = np.array([col1, col2]).T
    df = pd.DataFrame(data=data[1:, :], columns=data[0, :])
    print('total: {:.2f}s'.format(time.time() - t0))
    return df

def main():
    test_loader=DataLoader(CarDataSet(ROOT,TEST,trainable=False),batch_size=4)
    model=uNet(NUM_CLASS).cuda()
    model=load_model(model,args.ckpt)
    df=test(model,test_loader,os.path.join(ROOT,TEST_MASK))
    df.to_csv(os.path.join(ROOT,csv_fimename),index=False)

def load_model(model,ckpt):
    if os.path.isfile(ckpt):
        print('==> loading checkpoint {}'.format(ckpt))
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        print("==> loaded checkpoint '{}'".format(ckpt))
        return model
    else:
        print("==> no checkpoint found at '{}'".format(ckpt))


if __name__ == '__main__':
    main()
    
