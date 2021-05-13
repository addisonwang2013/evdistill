#!/usr/bin/python
# -*- encoding: utf-8 -*-


class Config(object):
    def __init__(self):
        ## model and loss_folder
        self.ignore_label = 255
        self.aspp_global_feature = False
        
## dataset
        self.n_classes = 6
        self.datapth = './data/'
        self.n_workers = 8
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.pool_scale = 0.8
        self.gpu_ids = [0, 1, 2]
       

        ## eval control
        self.data_path_test = './dataset/ddd17/'
        self.eval_batchsize = 1
        self.eval_n_workers = 2
        self.eval_scales = [1]
        self.eval_flip = False
        self.apsseg_ckpt ='./res/ddd17/ddd17_aps_ckpt.pth'
        self.eventseg_ckpt ='./res/ddd17/eventdual_2ch_iter_28000_event_0.58_best.pth'

        self.save_dir = './test_dir/ddd17/'

