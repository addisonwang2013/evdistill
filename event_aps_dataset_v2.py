#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform_event import *
from collections import namedtuple
import glob


# enable eager mode

#

class EventAPS_Dataset(Dataset):
    def __init__(self, cfg, mode='train', crop_size=(256, 512), *args, **kwargs):
        super(EventAPS_Dataset, self).__init__(*args, **kwargs)
        assert mode in ('train', 'other', 'test', 'val')

        self.mode = mode
        self.cfg = cfg
        self.crop_h, self.crop_w = crop_size[0], crop_size[1]
        self.imgs = {}
        imgnames = []
        impth = osp.join(cfg.data_path_test, 'images', mode)
        images = glob.glob(osp.join(impth, '*.png'))
        names = [osp.basename(el.split('.')[1]) for el in images]
        impths = images
        imgnames.extend(names)
        self.imgs.update(dict(zip(names, impths)))


        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(cfg.data_path_test, 'labels', mode)
        folders = os.listdir(gtpth)
        labels = glob.glob(osp.join(gtpth, '*.png'))
        lbnames = [osp.basename(el.split('.')[1]) for el in labels]
        lbpths = labels
        gtnames.extend(lbnames)
        self.labels.update(dict(zip(lbnames, lbpths)))
        self.imnames = imgnames
        self.len = len(self.imnames)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __getitem__(self, idx):
        fn = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth).convert("RGB")

        name = os.path.splitext(os.path.basename(impth))[0]

        # split AB image into A and B
        w, h = img.size
        w2 = int(w / 2)
        event = img.crop((0, 0, w2, h))  # crop entire image to get event
        aps = img.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        event = event.resize((self.crop_w, self.crop_h), Image.BICUBIC)  # resize event
        aps = aps.resize((self.crop_w, self.crop_h), Image.BICUBIC)

        label = Image.open(lbpth)
        label = label.resize((self.crop_w, self.crop_h), Image.NEAREST)

        event = self.to_tensor(event)
        aps = self.to_tensor(aps)
        label = np.array(label).astype(np.int64)[np.newaxis, :]

        return event, aps, label, name

    def __len__(self):
        return self.len

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    ds = EventAPS_Dataset('./dataset/eventdataset', mode='train')
    dl = DataLoader(ds,
                    batch_size=4,
                    shuffle=True,
                    num_workers=4,
                    drop_last=True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
