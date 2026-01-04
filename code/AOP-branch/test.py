import argparse
import os
import h5py
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import RDN
from utils import *
from parser import parser

if __name__ == '__main__':

    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = RDN(num_channels=4,
                num_features=16,
                growth_rate=16,
                num_blocks=12,
                num_layers=6
                ).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    file_list = os.listdir(args.image_file)
    for file in file_list:
        print(file)
        path = os.path.join(args.image_file,file)
        with h5py.File(path, 'r') as f:
            lr = np.array(f['Norm_photon']).T
    #
    # image_width = image.width #// args.scale) * args.scale
    # image_height = image.height# // args.scale) * args.scale
        #lr = image
    #hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    #lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    #bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    #bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

        lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) 
    #hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        lr = torch.from_numpy(lr).to(device)
    #hr = torch.from_numpy(hr).to(device)

        with torch.no_grad():
            preds = model(lr).squeeze(0)
        preds_numpy = preds.permute(1, 2, 0).cpu().numpy()
        
        print(f"Model output range: {np.min(preds_numpy)} to {np.max(preds_numpy)}")
        #output.save(args.image_file.replace('.', '_rdn.'))
        s0, dolp, aop, docp = cal_stokes_dolp(preds_numpy)

        preds_normalized = (preds_numpy - np.min(preds_numpy)) / (np.max(preds_numpy) - np.min(preds_numpy) + 1e-8)
        output = pil_image.fromarray(np.uint8(preds_normalized * 255))
        output_name = file.split('.')[0]
        save_depart_result(args, s0, dolp, aop, output_name, output, preds_numpy, docp)
    #preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
    #hr_y = convert_rgb_to_y(denormalize(hr.squeeze(0)), dim_order='chw')

    # preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
    # hr_y = hr_y[args.scale:-args.scale, args.scale:-args.scale]

    # psnr = calc_psnr(hr_y, preds_y)
    # print('PSNR: {:.2f}'.format(psnr))


