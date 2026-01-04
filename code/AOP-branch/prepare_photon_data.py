import argparse
import glob

import cv2
import h5py
import os
import numpy as np
import PIL.Image as pil_image
from torchvision.transforms import transforms
from scipy.io import loadmat, savemat  # 添加savemat导入

def streach(img):
    minval = np.percentile(img,20)

    out = (img-minval)/(np.nanmax(img)-minval)
    return out

if __name__ == '__main__':

    path = '/media/stokes/98743610-3181-40f5-a450-083cd5b35a16/stokes/LIuhedong/Code/denoise_mono/pytorch_unet/data_learn_picture'
    img = loadmat('data_learn_picture/1_250x244_80/1ms/CH2.png')
    img_input = np.zeros((img.shape[0],img.shape[1],4))
    img_label = np.zeros((img.shape[0], img.shape[1], 4))
    n=0
    for file in os.listdir(path):
        print(file)
        inputlist = os.listdir(os.path.join(path,file,'1ms'))
        inputlist = sorted(inputlist)
        labellist = os.listdir(os.path.join(path,file,'500ms'))
        labellist = sorted(labellist)

        for i in range(4):
            input = cv2.imread(os.path.join(path,file,'1ms',inputlist[i]),-1).astype(np.float32)/255
            label = cv2.imread(os.path.join(path, file,'500ms',labellist[i] ),-1).astype(np.float32)/255

            img_input[:,:,i] = input
            img_label[:,:,i] = label

        img_input = np.clip(img_input*255.0,0,255)
        img_label = np.clip(img_label * 255.0,0,255)

        input_out_path = 'data/train/input'
        label_out_path = 'data/train/label'
        os.makedirs(input_out_path,exist_ok=True)
        os.makedirs(label_out_path, exist_ok=True)
        cv2.imwrite(os.path.join(input_out_path,'{}.png'.format(n)),img_input.astype(np.uint8))
        cv2.imwrite(os.path.join(label_out_path,'{}.png'.format(n)), img_label.astype(np.uint8))
        n+=1