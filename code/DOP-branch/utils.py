import torch
import numpy as np
import cv2
import PIL.Image as pil_image
from scipy.io import savemat
def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def denormalize(img):
    img = img.mul(255.0).clamp(0.0, 255.0)
    return img


#def preprocess(img, device):
    # img = np.array(img).astype(np.float32)
    # ycbcr = convert_rgb_to_ycbcr(img)
    # x = ycbcr[..., 0]
    # x /= 255.
    # x = torch.from_numpy(x).to(device)
    # x = x.unsqueeze(0).unsqueeze(0)
    # return x, ycbcr


def calc_psnr(img1, img2, max=255.0):
    return 10. * ((max ** 2) / ((img1 - img2) ** 2).mean()).log10()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import os
def imsave(image, path):
    cv2.imwrite(os.path.join(os.getcwd(), path), image)


def save_merge_result(config, x, output_name):
    dofp = np.zeros((x.shape[0] * 2, x.shape[1] * 2))
    # print(dofp.shape)
    img0 = x[:, :, 0]
    img45 = x[:, :, 1]
    img90 = x[:, :, 2]
    img135 = x[:, :, 3]

    dofp[0:dofp.shape[0]:2, 0:dofp.shape[1]:2] = img90
    dofp[1:dofp.shape[0]:2, 0:dofp.shape[1]:2] = img135
    dofp[0:dofp.shape[0]:2, 1:dofp.shape[1]:2] = img45
    dofp[1:dofp.shape[0]:2, 1:dofp.shape[1]:2] = img0

    imsave(dofp, os.path.join(config.result_dir, "merge") + '/%s_dofp.png' % output_name)


def save_depart_result(config, s0, dolp, aop, output_name, output, img, docp):
    four_channels_dir = os.path.join(os.path.join(os.getcwd(), config.result_dir), "depart")
    # depart_channels_dir = os.path.join(os.path.join(os.getcwd(), config.result_dir), "depart")
    dolp_dir = os.path.join(os.path.join(os.getcwd(), config.result_dir), "depart", 'dolp')
    aop_dir = os.path.join(os.path.join(os.getcwd(), config.result_dir), "depart", 'aop')
    s0_dir = os.path.join(os.path.join(os.getcwd(), config.result_dir), "depart", 's0')
    mat_dir = os.path.join(os.path.join(os.getcwd(), config.result_dir), "depart", 'mat')
    docp_dir = os.path.join(os.path.join(os.getcwd(), config.result_dir), "depart", 'docp')
    if not os.path.isdir(four_channels_dir):
        os.makedirs(four_channels_dir)
    # if not os.path.isdir(depart_channels_dir):
    #     os.makedirs(depart_channels_dir)
    if not os.path.isdir(dolp_dir):
        os.makedirs(dolp_dir)

    if not os.path.isdir(aop_dir):
        os.makedirs(aop_dir)

    if not os.path.isdir(s0_dir):
        os.makedirs(s0_dir)

    if not os.path.isdir(mat_dir):
        os.makedirs(mat_dir)
    
    if not os.path.isdir(docp_dir):
        os.makedirs(docp_dir)
   
    # save_depart_channel_image = os.path.join(depart_channels_dir, output_name)
    # if not os.path.isdir(save_depart_channel_image):
    #     os.makedirs(save_depart_channel_image)
   
    ""
    img = np.array(img, dtype=np.float64)
    output = np.array(output, dtype=np.float64)
    #img =255*(img-np.nanmin(img))/(np.nanmax(img)-np.nanmin(img))
    ""
    #img = np.uint8(img)

    imsave_aop(aop, aop_dir, output_name)
    imsave_dolp(dolp, dolp_dir, output_name)
    imsave_s0(s0, s0_dir, output_name)

    imsave_fourchannel(output, four_channels_dir, output_name)
    imsave_mat(img, mat_dir, output_name)
    imsave_docp(docp, docp_dir, output_name)
    # imsave_s0_dolp(s0,dolp,aop,save_depart_channel_image,output_name)
    ###################################修改修改修改########################
    # name = [0,45,90,135]
    # channels = x.shape[2]#四个通道
    # for channel in range(channels):
    #     cv2.imwrite(os.path.join(save_depart_channel_image, "{}.png").format(name[channel]), x[:, :, channel])

import  math
def cal_stokes_dolp(img):
    img = np.array(img, dtype=np.float64)
    
    print(f"cal_stokes_dolp input range: {np.min(img)} to {np.max(img)}")

    img0 = img[:, :, 0]
    img90 = img[:, :, 1]
    img_circule = img[:, :, 2]
    img135 = img[:, :, 3]

    S0 = img0 + img90
    S1 = img0 - img90
    S2 = img0 + img90 - img135 * 2
    S3=2*img_circule-(img90 + img0)

    DoLP = np.sqrt((S1 ** 2 + S2 ** 2) / (S0 + 0.0001) ** 2)

    def normalize_to_255(data, name):
        data_min, data_max = np.min(data), np.max(data)
        print(f"{name} raw range: {data_min} to {data_max}")
        
        if name == "DoLP" or name == "AoP" or name == "DoCP":
           
            if data_max > data_min and data_max > 1e-6:
                p5, p95 = np.percentile(data, [5, 95])
                if p95 > p5:
                    normalized = 255.0 * (data - p5) / (p95 - p5)
                    normalized = np.clip(normalized, 0, 255)
                else:
                    normalized = np.full_like(data, 128.0)
            else:
                normalized = np.zeros_like(data)
        else:
            
            if data_max > data_min:
                normalized = 255.0 * (data - data_min) / (data_max - data_min)
            else:
                normalized = np.zeros_like(data)
        
        result = np.clip(normalized, 0, 255).astype(np.uint8)
        print(f"{name} final range: {np.min(result)} to {np.max(result)}")
        return result
    DoLP = normalize_to_255(DoLP, "DoLP") 
      
    AoP = 0.5 * np.arctan2(S2, S1)
    AoP = (AoP + math.pi / 2) / math.pi 
    AoP = normalize_to_255(AoP, "AoP")
    DoCP=np.abs(S3)/S0
    DoCP = normalize_to_255(DoCP, "DoCP")
    
    S0 = normalize_to_255(S0, "S0")


    return S0, DoLP, AoP, DoCP

def cal_stokes_input_dolp(img):
    img = np.array(img, dtype=np.float64)

    print(f"cal_stokes_input_dolp input range: {np.min(img)} to {np.max(img)}")

    img0 = img[:, :, 0]
    img90 = img[:, :, 1]
    img_circule = img[:, :, 2]
    img135 = img[:, :, 3]

    S0 = img0 + img90
    S1 = img0 - img90
    S2 = img0 + img90 - img135 * 2
    S3 = 2 * img_circule - (img90 + img0)

    
    print(f"S0 range: {np.min(S0)} to {np.max(S0)}")
    print(f"S3 range: {np.min(S3)} to {np.max(S3)}")

    DoLP = np.sqrt((S1 ** 2 + S2 ** 2) / (S0 + 0.0001) ** 2)

    def normalize_to_255(data, name):
        data_min, data_max = np.min(data), np.max(data)
        print(f"{name} raw range: {data_min} to {data_max}")

        if name == "DoLP" or name == "AoP" or name == "DoCP":
           
            if data_max > data_min and data_max > 1e-6:
                p5, p95 = np.percentile(data, [5, 90])
                print(f"{name} percentile range: {p5} to {p95}")
                if p95 > p5:
                    normalized = 255.0 * (data - p5) / (p95 - p5)
                    normalized = np.clip(normalized, 0, 255)
                else:
                    normalized = np.full_like(data, 128.0)
            else:
                print(f"Warning: {name} has very small range, setting to zeros")
                normalized = np.zeros_like(data)
        else:
            
            if data_max > data_min:
                normalized = 255.0 * (data - data_min) / (data_max - data_min)
            else:
                normalized = np.zeros_like(data)

        result = np.clip(normalized, 0, 255).astype(np.uint8)
        print(f"{name} final range: {np.min(result)} to {np.max(result)}")
        return result

    DoLP = normalize_to_255(DoLP, "DoLP")

    
    AoP = 0.5 * np.arctan2(S2, S1)
    AoP = (AoP + math.pi / 2) / math.pi  # 归一化到0-1
    AoP = normalize_to_255(AoP, "AoP")

    
    safe_S0 = np.where(np.abs(S0) < 1e-6, 1e-6, S0)
    
    DoCP_raw = np.abs(S3 / safe_S0)

    
    print(f"DoCP raw calculation range: {np.min(DoCP_raw)} to {np.max(DoCP_raw)}")
    print(f"DoCP raw mean: {np.mean(DoCP_raw)}")
    print(f"DoCP raw std: {np.std(DoCP_raw)}")

    
    if np.max(DoCP_raw) < 0.01:
        print("DoCP values are very small, using alternative normalization")
        
        p1, p99 = np.percentile(DoCP_raw, [0, 99])
        if p99 > p1:
            DoCP = 255.0 * (DoCP_raw - p1) / (p99 - p1)
            DoCP = np.clip(DoCP, 0, 255).astype(np.uint8)
        else:
           
            DoCP = np.clip(DoCP_raw * 10000, 0, 255).astype(np.uint8)
    else:
        DoCP = normalize_to_255(DoCP_raw, "DoCP")

    
    S0 = normalize_to_255(S0, "S0")

    return S0, DoLP, AoP, DoCP
# def imsave_s0_dolp(s0,dolp,aop,path,name):
#     cv2.imwrite(os.path.join(path,"{}_s0.png".format(name)),s0)
#     cv2.imwrite(os.path.join(path,"{}_dolp.png".format(name)),dolp)
#     cv2.imwrite(os.path.join(path, "{}_aop.png".format(name)),aop)
def imsave_aop(aop, path, name):
    cv2.imwrite(os.path.join(path, "{}_aop.png".format(name)), aop)
    #aop.save(os.path.join(path, "{}_aop.png".format(name)))


def imsave_dolp(dolp, path, name):
    cv2.imwrite(os.path.join(path, "{}_dolp.png".format(name)), dolp)
    #dolp.save(os.path.join(path, "{}_dolp.png".format(name)))


def imsave_s0(s0, path, name):
    cv2.imwrite(os.path.join(path, "{}_s0.png".format(name)), s0)
    #s0.save(os.path.join(path, "{}_s0.png".format(name)))

def imsave_fourchannel(img, path, name):
    cv2.imwrite(os.path.join(path, "{}_s0.png".format(name)), img)
    #s0.save(os.path.join(path, "{}_s0.png".format(name)))

def imsave_mat(img, path, name):
    savemat(os.path.join(path, "{}_.mat".format(name)), {'output': img})
    #s0.save(os.path.join(path, "{}_s0.png".format(name)))

def imsave_docp(docp, path, name):
    cv2.imwrite(os.path.join(path, "{}_docp.png".format(name)), docp)
    #s0.save(os.path.join(path, "{}_s0.png".format(name)))

def imsave_noise_map(img, name):
    noise_map = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    img0 = img[:, :, 0]
    img45 = img[:, :, 1]
    img90 = img[:, :, 2]
    img135 = img[:, :, 3]
    noise_map[0:noise_map.shape[0]:2, 0:noise_map.shape[1]:2] = img90
    noise_map[1:noise_map.shape[0]:2, 0:noise_map.shape[1]:2] = img135
    noise_map[0:noise_map.shape[0]:2, 1:noise_map.shape[1]:2] = img45
    noise_map[1:noise_map.shape[0]:2, 1:noise_map.shape[1]:2] = img0

    # noise_map = noise_map*(1/np.nanmax(noise_map))
    noise_map = np.clip(noise_map * 255, 0, 255)
    noise_map_1 = noise_map * 30
    path = os.path.join(os.getcwd(), 'residual_map')
    if not os.path.isdir(path):
        os.mkdir(path)
    cv2.imwrite('residual_map' + '/' + '{}.png'.format(name), noise_map)
    cv2.imwrite('residual_map' + '/' + '{}_1.png'.format(name), noise_map_1)


from math import cos, sin, radians


def new_i0(i0, i1, i2):
    s0 = 2 / 3 * (i0 + i1 + i2)
    s1 = i0 + i2 - 2 * i1
    s2 = i0 - i2
    new = 0.5 * (s0 + s1)
    return new


def new_i45(i0, i1, i2):
    s0 = 2 / 3 * (i0 + i1 + i2)
    s1 = i0 - i1
    s2 = i0 + i1 - 2 * i2
    new = 0.5 * (s0 + s2)
    return new


def new_i90(i0, i1, i2):
    s0 = 2 / 3 * (i0 + i1 + i2)
    s1 = 2 * i0 - i1 - i2
    s2 = i1 - i2
    new = 0.5 * (s0 - s1)
    return new


def new_i135(i0, i1, i2):
    s0 = 2 / 3 * (i0 + i1 + i2)
    s1 = i0 - i2
    s2 = 2 * i1 - i0 - i2
    new = 0.5 * (s0 - s2)
    return new


def cal_stokes_dolp_3c(img):
    img = np.array(img)
    img0 = img[:, :, 0]
    img45 = img[:, :, 1]
    img90 = img[:, :, 2]

    S0 = (img0.astype(np.float32) + img45.astype(np.float32) +
          img90.astype(np.float32)) * 2 / 3
    S1 = img0.astype(np.float32) - img90.astype(np.float32)
    S2 = 2 * img45.astype(np.float32) - img0.astype(np.float32) - img90.astype(np.float32)
    # 归一化在255内
    DoLP = np.sqrt((S1 ** 2 + S2 ** 2) / (S0 + 0.00001) ** 2)
    DoLP = DoLP * (1 / np.nanmax(DoLP))  
    DoLP = np.clip(DoLP * 255, 0, 255)  
    AoP = 1 / 2 * np.arctan2(S2, S1)
    AoP = (AoP + math.pi / 2) / math.pi
    AoP = AoP * (1 / np.nanmax(AoP))  
    AoP = np.clip(AoP * 255, 0, 255)  
    S0 = S0 * (255 / np.nanmax(S0))
    return S0, DoLP, AoP


def normalize_to_uint8(img):

   
    img = np.float32(img)
    
    
    min_val = np.nanmin(img)
    max_val = np.nanmax(img)
    
   
    if max_val == min_val:
        normalized = np.zeros_like(img)
    else:
        
        normalized = (img - min_val) / (max_val - min_val)
    
    
    normalized = normalized * 255.0
    
    
    normalized = np.clip(normalized, 0, 255)
    
    
    return normalized.astype(np.float32)


