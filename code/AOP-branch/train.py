import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from models import RDN
#from unet import UNet
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr, convert_rgb_to_y, denormalize
import math

class CombinedLoss(nn.Module):
    def __init__(self, l1_weight, aop_weight):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l1_weight = l1_weight
        self.aop_weight = aop_weight
    
    def calculate_aop(self, img):

        img0 = img[:, 0:1, :, :]    
        img90 = img[:, 1:2, :, :]   
        img_circle = img[:, 2:3, :, :] 
        img135 = img[:, 3:4, :, :]  
        
        
        S0 = img0 + img90
        S1 = img0 - img90
        S2 = img0 + img90 - img135 * 2
        
        
        AoP = 0.5 * torch.atan2(S2, S1)
        AoP = (AoP + math.pi / 2) / math.pi  # 归一化到0-1
        
        return AoP
    
    def forward(self, pred, target):
        
        l1_loss = self.l1_loss(pred, target)
        
        
        pred_aop = self.calculate_aop(pred)
        target_aop = self.calculate_aop(target)
        
        
        aop_loss = self.l1_loss(pred_aop, target_aop)
        
        
        total_loss = self.l1_weight * l1_loss + self.aop_weight * aop_loss
        
        return total_loss, l1_loss, aop_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='data/train.h5')
    parser.add_argument('--eval-file', type=str, default='data/eval.h5')
    parser.add_argument('--outputs-dir', type=str, default='checkpoint')
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--num-features', type=int, default=32)
    parser.add_argument('--growth-rate', type=int, default=32)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=6)
    #parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--l1-weight', type=float, default=1, help='Weight for L1 loss')
    parser.add_argument('--aop-weight', type=float, default=0, help='Weight for AOP loss')
    args = parser.parse_args()

    #args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = RDN(num_channels=4,
                num_features=16,
                growth_rate=16,
                num_blocks=12,
                num_layers=6
                ).to(device)
    #UNet(in_channels=4,out_channels=4).to(device)

    # if args.weights_file is not None:
    #     state_dict = model.state_dict()
    #     for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
    #         if n in state_dict.keys():
    #             state_dict[n].copy_(p)
    #         else:
    #             raise KeyError(n)

    criterion = CombinedLoss(l1_weight=args.l1_weight, aop_weight=args.aop_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (0.1 ** (epoch // int(args.num_epochs * 0.8)))

        model.train()
        epoch_losses = AverageMeter()
        epoch_l1_losses = AverageMeter()
        epoch_aop_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=120) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                
                total_loss, l1_loss, aop_loss = criterion(preds, labels)

                epoch_losses.update(total_loss.item(), len(inputs))
                epoch_l1_losses.update(l1_loss.item(), len(inputs))
                epoch_aop_losses.update(aop_loss.item(), len(inputs))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                t.set_postfix(totaloss='{:.6f}'.format(epoch_losses.avg), 
                             l1='{:.6f}'.format(epoch_l1_losses.avg),
                             aop='{:.6f}'.format(epoch_aop_losses.avg))
                t.update(len(inputs))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs)

            #preds = convert_rgb_to_y(denormalize(preds.squeeze(0)), dim_order='chw')
            #labels = convert_rgb_to_y(denormalize(labels.squeeze(0)), dim_order='chw')

            # preds = preds[args.scale:-args.scale, args.scale:-args.scale]
            # labels = labels[args.scale:-args.scale, args.scale:-args.scale]

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
