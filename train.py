# python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/27 15:31
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2022 Liupeng Lin. All Rights Reserved.


import os, datetime, time
import h5py
import glob
import re
import warnings
import argparse
import numpy as np
import torch
from cafe import CAFE
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dataset import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import gc
from loss import *
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='CAFE')
parser.add_argument('--model', default='CAFE', type=str, help='choose path of model')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--epoch', default=300, type=int, help='number of train epoch')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')
parser.add_argument('--train_data', default='data/h5/train_cafe_250_500_20221017.h5', type=str, help='path of train data')
parser.add_argument('--gpu', default='0,1', type=str, help='gpu id')
parser.add_argument('--nDens', type=int, default=5, help='nDenselayer of RDB')
parser.add_argument('--GrowRate', type=int, default=32, help='growthRate of dense connection block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--ncha_landsat', type=int, default=6, help='number of hr channels to use')
parser.add_argument('--ncha_modis', type=int, default=6, help='number of lr channels to use')
args = parser.parse_args()


cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

batch_size = args.batch_size
nepoch = args.epoch
savedir = os.path.join('model', args.model)

if not os.path.exists(savedir):
    os.mkdir(savedir)

log_header = [
    'epoch',
    'iteration',
    'train/loss',
]
if not os.path.exists(os.path.join(savedir, 'log.csv')):
    with open(os.path.join(savedir, 'log.csv'), 'w') as f:
        f.write(','.join(log_header) + '\n')


def find_checkpoint(savedir):
    file_list = glob.glob(os.path.join(savedir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for m in file_list:
            result = re.findall(".*model_(.*).pth.*", m)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def main():
    # dataset generator
    print("==> Generating data")
    hf = h5py.File(args.train_data, 'r+')
    modis_tar = np.float32(hf['data1'])
    modis_ref = np.float32(hf['data2'])
    landsat_ref = np.float32(hf['data3'])
    landsat_tar = np.float32(hf['label'])

    modis_tar = torch.from_numpy(modis_tar).view(-1, args.ncha_modis, 40, 40)
    modis_ref = torch.from_numpy(modis_ref).view(-1, args.ncha_modis, 40, 40)
    landsat_ref = torch.from_numpy(landsat_ref).view(-1, args.ncha_landsat, 40, 40)
    landsat_tar = torch.from_numpy(landsat_tar).view(-1, args.ncha_landsat, 40, 40)

    train_set = CAFEDataset(modis_tar, modis_ref, landsat_ref, landsat_tar)
    train_loader = DataLoader(dataset=train_set, num_workers=8, drop_last=True, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    print("==> Building model")
    model = CAFE(args)
    criterion = CELoss()

    if cuda:
        print("==> Setting GPU")
        print("===> gpu id: '{}'".format(args.gpu))
        model = model.cuda()
        criterion = criterion.cuda()

    print("==> Setting optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr,  weight_decay=0)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # learning rates adjust

    initial_epoch = find_checkpoint(savedir=savedir)
    if initial_epoch > 0:
        print('==> Resuming by loading epoch %03d' % initial_epoch)
        state_dict = torch.load(os.path.join(savedir, 'model_%03d.pth' % initial_epoch))
        model.load_state_dict(state_dict)

    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model)

    for epoch in range(initial_epoch, nepoch):
        scheduler.step(epoch)
        start_time = time.time()

        # train
        model.train()
        for iteration, batch in enumerate(train_loader):
            modis_tar_batch, modis_ref_batch, landsat_ref_batch, landsat_tar_batch = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
            if cuda:
                modis_tar_batch = modis_tar_batch.cuda()
                modis_ref_batch = modis_ref_batch.cuda()
                landsat_ref_batch = landsat_ref_batch.cuda()
                landsat_tar_batch = landsat_tar_batch.cuda()
            optimizer.zero_grad()
            out = model(modis_tar_batch, modis_ref_batch, landsat_ref_batch)
            loss = criterion(out, landsat_tar_batch)

            print('%4d %4d / %4d loss = %2.6f' % (epoch + 1, iteration, train_set.landsat_tar.size(0)/batch_size, loss.data))
            loss.backward()
            optimizer.step()
            with open(os.path.join(savedir, 'log.csv'), 'a') as file:
                log = [epoch, iteration] + [loss.data.item()]
                log = map(str, log)
                file.write(','.join(log) + '\n')
        if len(args.gpu) > 1:
            if ((epoch + 1) % 5) == 0:
                torch.save(model.module.state_dict(), os.path.join(savedir, 'model_%03d.pth' % (epoch + 1)))
        else:
            if ((epoch + 1) % 5) == 0:
                torch.save(model.state_dict(), os.path.join(savedir, 'model_%03d.pth' % (epoch + 1)))
        gc.collect()
        elapsed_time = time.time() - start_time
        print('epcoh = %4d , time is %4.4f s' % (epoch + 1, elapsed_time))


if __name__ == '__main__':
    main()
