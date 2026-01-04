import argparse
import glob

import cv2
import h5py
import os
import numpy as np
import PIL.Image as pil_image
from torchvision.transforms import transforms
from scipy.io import loadmat

def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    input_list = sorted(glob.glob('{}/*'.format(args.input_dir)))
    patch_idx = 0

    for i, input_path in enumerate(input_list):
        label_path = input_path.replace('input','label')
        # hr = pil_image.open(label_path)
        # lr = pil_image.open(input_path)
        try:
        
            hr_data = loadmat(label_path)
            lr_data = loadmat(input_path)
            lr = lr_data['Norm_photon']
            hr = hr_data['Norm_photon']
        except NotImplementedError:
            
            with h5py.File(label_path, 'r') as f:
              
                hr = np.array(f['Norm_photon']).T
            
            with h5py.File(input_path, 'r') as f:
                lr = np.array(f['Norm_photon']).T

        if len(lr.shape) == 3:
            h, w, c = lr.shape
        else:
            h,w = lr.shape
        for x in range(0, h - args.patch_size + 1, args.stride):
            for y in range(0, w - args.patch_size + 1, args.stride):
                sub_lr = lr[x:x + args.patch_size, y:y + args.patch_size]
                sub_hr = hr[x:x + args.patch_size, y:y + args.patch_size]
                lr_group.create_dataset(str(patch_idx), data=sub_lr)
                hr_group.create_dataset(str(patch_idx), data=sub_hr)
                patch_idx += 1
        #print(hr.mode)
        # hr_crop = transforms.FiveCrop(size=(hr.height // 2, hr.width // 2))(hr)
        # lr_crop = transforms.FiveCrop(size=(hr.height // 2, hr.width // 2))(lr)
        # print(len(hr_crop),len(lr_crop))
        #
        # for i in range(5):
        #     hr = np.array(hr_crop[i])
        #     lr = np.array(lr_crop[i])
        #     print(hr.shape)
        #     lr_group.create_dataset(str(patch_idx), data=lr)
        #     hr_group.create_dataset(str(patch_idx), data=hr)
        #
        #     patch_idx += 1
        print(i, patch_idx, label_path)
    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.eval_output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, input_path in enumerate(sorted(glob.glob('{}/*'.format(args.eval_input_dir)))):

        label_path  = input_path.replace('input','label')
       
        try:
           
            lr_data = loadmat(input_path)
            hr_data = loadmat(label_path)
            lr = lr_data['Norm_photon']
            hr = hr_data['Norm_photon']
            
        except NotImplementedError:
            
            with h5py.File(input_path, 'r') as f:
               
                lr = np.array(f['Norm_photon']).T
            
            with h5py.File(label_path, 'r') as f:
                hr = np.array(f['Norm_photon']).T

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

        print(i)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='/media/ti/WD/SHR/rdn_modelzhen - AOP-sunshihanshu/data/train/input')
    parser.add_argument('--label-dir', type=str, default='/media/ti/WD/SHR/rdn_modelzhen - AOP-sunshihanshu/data/train/label')
    parser.add_argument('--output-path', type=str, default='data/train.h5')

    parser.add_argument('--eval-input-dir', type=str,
                        default='data/eval/input')
    parser.add_argument('--eval-label-dir', type=str,
                        default='data/eval/label')
    parser.add_argument('--eval-output-path', type=str, default='data/eval.h5')

    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)

    args = parser.parse_args()


    train(args)

    eval(args)