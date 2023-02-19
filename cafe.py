# python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/27 15:11
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2022 Liupeng Lin. All Rights Reserved.


import torch
from torch import nn
import torch.nn.init as init


class Dense(nn.Module):
    """
    Dense connection  
    """
    def __init__(self, nFeat, GrowRate):
        super(Dense, self).__init__()
        self.conv_dense = nn.Sequential(nn.Conv2d(nFeat, GrowRate, kernel_size=3, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        out1 = self.conv_dense(x)
        out = torch.cat((x, out1), 1)
        return out


class RDB(nn.Module):
    """
    Residual Dense connection Block 
    """
    def __init__(self, nFeat, nDens, GrowRate):
        super(RDB, self).__init__()
        nFeat_ = nFeat
        modules = []
        for i in range(nDens):
            modules.append(Dense(nFeat_, GrowRate))
            nFeat_ += GrowRate
            self.dense_layers = nn.Sequential(*modules)
            self.conv_1x1 = nn.Conv2d(nFeat_, nFeat, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out1 = self.conv_1x1(self.dense_layers(x))
        out = torch.add(x, out1)
        return out


class CA(nn.Module):
    """
    Channel Attention
    """
    def __init__(self, nFeat, ratio=8):
        super(CA, self).__init__()
        self.ca_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.ca_conv2 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=1, padding=0, bias=True), nn.PReLU())
        self.ca_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat // ratio, 1, padding=0, bias=True), nn.PReLU())
        self.ca_fc2 = nn.Sequential(
            nn.Conv2d(nFeat // ratio, nFeat, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        ca_weight_avg = self.ca_fc2(self.ca_fc1(self.ca_avg_pool(x)))
        out = self.ca_conv2(torch.mul(x, ca_weight_avg))
        return out


class SA(nn.Module):
    """
    Spatial Attention
    """
    def __init__(self, nFeat):
        super(SA, self).__init__()
        self.sa_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, groups=nFeat, bias=True), nn.PReLU())
        self.sa_conv1x1 = nn.Sequential(
            nn.Conv2d(nFeat, 1, kernel_size=1, padding=0, bias=True), nn.Sigmoid())
        self.sa_conv2 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        sa_weight = self.sa_conv1x1(self.sa_conv1(x))
        out = self.sa_conv2(x * sa_weight)
        return out


class CSA(nn.Module):
    def __init__(self, nFeat):
        super(CSA, self).__init__()
        self.csa_ca = CA(nFeat)
        self.csa_sa = SA(nFeat)
        self.csa_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        out1 = self.csa_ca(x)
        out2 = self.csa_sa(x)
        out = self.csa_conv1(torch.add(out1, out2))
        return out


class RCSA(nn.Module):
    def __init__(self, nFeat):
        super(RCSA, self).__init__()
        self.rcsa_ca = CA(nFeat)
        self.rcsa_sa = SA(nFeat)
        self.rcsa_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        out1 = self.rcsa_ca(x)
        out2 = self.rcsa_sa(x)
        out = torch.add(x, self.rcsa_conv1(torch.add(out1, out2)))
        return out


class SBCA(nn.Module):
    """
    Single Band Cross Attention
    """
    def __init__(self, nFeat):
        super(SBCA, self).__init__()
        nFeat_ = nFeat//2
        # nFeat_ = nFeat
        self.sbca_conv1 = nn.Sequential(
            nn.Conv2d(1, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.sbca_conv2 = nn.Sequential(
            nn.Conv2d(1, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.sbca_conv3 = nn.Sequential(
            nn.Conv2d(nFeat_, nFeat_, kernel_size=3, padding=1, bias=True), nn.Sigmoid())
        self.sbca_conv4 = nn.Sequential(
            nn.Conv2d(nFeat_, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.sbca_conv1x1 = nn.Sequential(
            nn.Conv2d(nFeat_, 1, kernel_size=1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, lr, hr):
        lr_fea = self.sbca_conv1(lr)
        hr_fea = self.sbca_conv2(hr)
        out = self.sbca_conv4(torch.add(torch.mul(lr_fea, self.sbca_conv1x1(hr_fea)), torch.mul(hr_fea, self.sbca_conv3(lr_fea))))
        return out


class MSCA(nn.Module):
    """
    Multi-channel Separated Cross Attention
    """
    def __init__(self, args):
        super(MSCA, self).__init__()
        nFeat = args.nFeat
        nFeat_ = nFeat // 2
        ncha_data = args.ncha_modis
        self.msca_layers = nn.ModuleList()
        for i in range(ncha_data):
            self.msca_layers.append(SBCA(nFeat))
        self.msca_conv1 = nn.Sequential(
            nn.Conv2d(nFeat_*ncha_data, nFeat, kernel_size=3, padding=1, bias=True), nn.PReLU())

    def forward(self, mlr, mhr):
        num = mlr.shape[1]
        for i in range(num):
            out = self.msca_layers[i](torch.index_select(mlr, 1, torch.LongTensor([i]).cuda()), torch.index_select(mhr, 1, torch.LongTensor([i]).cuda()))
            if i == 0:
                out_ = out
            elif i > 0:
                out_ = torch.cat((out, out_), 1)
        out = self.msca_conv1(out_)
        return out


class FJCA(nn.Module):
    """
    Full-feature Joint Cross Attention
    """
    def __init__(self, nFeat):
        super(FJCA, self).__init__()
        self.fjca_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.fjca_conv2 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.fjca_conv3 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=1, padding=0, bias=True), nn.Sigmoid())
        self.fjca_conv4 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.fjca_conv1x1 = nn.Sequential(
            nn.Conv2d(nFeat, 1, kernel_size=1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, lr, hr):
        lr_fea = self.fjca_conv1(lr)
        hr_fea = self.fjca_conv2(hr)
        out = self.fjca_conv4(torch.add(torch.mul(lr_fea, self.fjca_conv1x1(hr_fea)), torch.mul(hr_fea, self.fjca_conv3(lr_fea))))
        return out


class ARDM(nn.Module):
    """
    Attention based Residual Dense Module
    """
    def __init__(self, args):
        super(ARDM, self).__init__()
        nDens = args.nDens
        nFeat = args.nFeat
        GrowRate = args.GrowRate
        self.ardm_rdb1 = RDB(nFeat, nDens, GrowRate)
        self.ardm_rdb2 = RDB(nFeat, nDens, GrowRate)
        self.ardm_rdb3 = RDB(nFeat, nDens, GrowRate)
        self.ardm_rcsa1 = RCSA(nFeat)
        self.ardm_conv1x1_1 = nn.Sequential(
            nn.Conv2d(nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True), nn.PReLU())

    def forward(self, x):
        out1 = self.ardm_rdb1(x)
        out2 = self.ardm_rdb2(out1)
        out3 = self.ardm_rdb3(out2)
        out = self.ardm_rcsa1(self.ardm_conv1x1_1(torch.cat((out1, out2, out3), 1)))
        return out


class CAFE(nn.Module):
    """
    Cross attention based Adaptive weight Fusion nEtwork
    """
    def __init__(self, args):
        super(CAFE, self).__init__()
        ncha_landsat = args.ncha_landsat
        ncha_modis = args.ncha_modis
        nFeat = args.nFeat

        self.cafe_msca1 = MSCA(args)
        self.cafe_msca2 = MSCA(args)

        self.cafe_fjca1 = FJCA(nFeat)

        self.cafe_ardm1 = ARDM(args)
        self.cafe_ardm2 = ARDM(args)
        self.cafe_ardm3 = ARDM(args)
        self.cafe_ardm4 = ARDM(args)

        self.cafe_conv1 = nn.Sequential(
            nn.Conv2d(ncha_modis, nFeat, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.cafe_conv2 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True), nn.Tanh())
        self.cafe_conv3 = nn.Sequential(
            nn.Conv2d(ncha_landsat, nFeat, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.cafe_conv4 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.cafe_conv5 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.cafe_conv6 = nn.Sequential(
            nn.Conv2d(nFeat, ncha_landsat, kernel_size=3, padding=1, bias=True))

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, modis_tar, modis_ref, landsat_ref):
        # temporal difference feature maps
        diff_fea = self.cafe_conv2(self.cafe_ardm1(self.cafe_conv1(torch.sub(modis_tar, modis_ref))))

        # landsat feature maps
        lan_fea = self.cafe_conv3(landsat_ref)

        # branch 1
        out1 = self.cafe_ardm2(self.cafe_msca1(modis_tar, landsat_ref))
        out2 = self.cafe_fjca1(out1, lan_fea)
        out3 = self.cafe_conv4(torch.add(out2, lan_fea))

        # branch 2
        out4 = self.cafe_ardm3(self.cafe_msca2(modis_ref, landsat_ref))
        out5 = self.cafe_fjca1(out4, lan_fea)
        out6 = self.cafe_conv5(torch.add(out5, lan_fea))

        # Adaptive Temporal Difference Weighting Mechanism
        out7 = self.cafe_ardm4(torch.add(torch.mul(out3, diff_fea), torch.mul(out6, diff_fea)))
        out = self.cafe_conv6(out7)

        return out
