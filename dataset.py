# python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/27 14:56
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2022 Liupeng Lin. All Rights Reserved.


from torch.utils.data import Dataset


class CAFEDataset(Dataset):
    def __init__(self, modis_tar, modis_ref, landsat_ref, landsat_tar):
        super(CAFEDataset, self).__init__()
        self.modis_tar = modis_tar
        self.modis_ref = modis_ref
        self.landsat_ref = landsat_ref
        self.landsat_tar = landsat_tar

    def __getitem__(self, index):
        batch_modis_tar = self.modis_tar[index]
        batch_modis_ref = self.modis_ref[index]
        batch_landsat_ref = self.landsat_ref[index]
        batch_landsat_tar = self.landsat_tar[index]
        return batch_modis_tar.float(), batch_modis_ref.float(), batch_landsat_ref.float(), batch_landsat_tar.float()

    def __len__(self):
        return self.landsat_tar.size(0)
