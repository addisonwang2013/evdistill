#!/usr/bin/python
# -*- encoding: utf-8 -*-
from modeling.deeplab import *
from models.deeplabv3plus import Deeplab_v3plus
from event_aps_dataset_v2 import EventAPS_Dataset
from evaluate import MscEval
from configs import config_factory
import torch
from torch.utils.data import DataLoader

cfg = config_factory['eventDistill_config']
ds = EventAPS_Dataset(cfg, mode='test')
dl = DataLoader(ds,
                batch_size = 1,
                shuffle = True,
                sampler = None,
                num_workers = cfg.n_workers,
                pin_memory = True,
                drop_last = True)

# event model
eventNet = DeepLab(backbone='resnet', num_classes=6)
eventNet.load_state_dict(torch.load(cfg.eventseg_ckpt))
eventNet.eval()
eventNet.cuda()


## APS model
net = DeepLab(backbone='resnet',num_classes=6)
net.load_state_dict(torch.load(cfg.apsseg_ckpt))
net.eval()
net.cuda()


evaluator = MscEval(cfg)
miou, miou_aps = evaluator(eventNet,  net)
print('miou of aps frames:%f; miou_of events:%f' %(miou_aps, miou))
