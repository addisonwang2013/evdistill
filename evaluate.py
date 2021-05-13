#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import *
from models.deeplabv3plus import Deeplab_v3plus
from configs import config_factory
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from event_aps_dataset_v2 import EventAPS_Dataset
import sys
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import argparse

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = -1,
            )
    return parse.parse_args()


class MscEval(object):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg

        self.distributed = None#dist.is_initialized()
        ## dataloader
        dsval = EventAPS_Dataset(cfg, mode='test')
        sampler = None
        if self.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dsval)
        self.dl = DataLoader(dsval,
                        batch_size = cfg.eval_batchsize,
                        sampler = sampler,
                        shuffle = False,
                        num_workers = cfg.eval_n_workers,
                        drop_last = False)

    def __call__(self, net, net2):
        ## evaluate
        n_classes = self.cfg.n_classes
        ignore_label = self.cfg.ignore_label

        hist_size = (self.cfg.n_classes, self.cfg.n_classes)
        hist = np.zeros(hist_size, dtype=np.float32)
        hist_aps = np.zeros(hist_size, dtype=np.float32)

        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(self.dl)
        else:
            diter = enumerate(tqdm(self.dl))
        for i, (imgs, aps, label, name) in diter:
            label = label.squeeze(1).cuda()

            N, H, W = label.shape

            probs = torch.zeros((N, n_classes, H, W)).cuda()
            probs_aps = torch.zeros((N, n_classes, H, W)).cuda()

            probs.requires_grad = False
            probs_aps.requires_grad = False

            for sc in self.cfg.eval_scales:
                new_hw = [int(H * sc), int(W * sc)]
                with torch.no_grad():
                    im = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)
                    aps = F.interpolate(aps, new_hw, mode='bilinear', align_corners=True)
                    im = im.cuda()
                    aps = aps.cuda()

                    out = net(im)[0]
                    out_aps = net2(aps)[0]
                    out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
                    out_aps = F.interpolate(out_aps, (H, W), mode='bilinear', align_corners=True)

                    prob = F.softmax(out, 1)
                    prob_aps = F.softmax(out_aps,1)

                    probs_aps+= prob_aps
                    probs += prob
                    if self.cfg.eval_flip:
                        out = net(torch.flip(im, dims=(3,)))
                        out_aps = net(torch.flip(aps, dims=(3,)))

                        out = torch.flip(out, dims=(3,))
                        out = F.interpolate(out, (H, W), mode='bilinear',
                                            align_corners=True)
                        prob = F.softmax(out, 1)
                        probs += prob

                        out_aps = torch.flip(out_aps, dims=(3,))
                        out_aps = F.interpolate(out_aps, (H, W), mode='bilinear',
                                            align_corners=True)
                        prob_aps = F.softmax(out_aps, 1)
                        probs_aps += prob_aps

                    del out, prob, prob_aps
            torch.cuda.empty_cache()
            preds = np.argmax(probs.cpu().numpy(), axis=1)
            preds_aps = np.argmax(probs_aps.cpu().numpy(), axis=1)
            label = label.cpu().numpy()
            keep = label != ignore_label
            hist_once = np.bincount(label[keep] * n_classes + preds[keep], minlength=n_classes ** 2).reshape(n_classes,
                                                                                                             n_classes)
            hist_once_aps = np.bincount(label[keep] * n_classes + preds_aps[keep], minlength=n_classes ** 2).reshape(n_classes,
                                                                                                             n_classes)
            hist = hist + hist_once
            hist_aps = hist_aps + hist_once_aps

        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
            dist.all_reduce(hist_aps, dist.ReduceOp.SUM)

        ious = np.diag(hist)/(hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        ious_aps = np.diag(hist_aps)/(hist_aps.sum(axis=1) + hist_aps.sum(axis=0) - np.diag(hist_aps))

        miou = np.nanmean(ious)
        miou_aps = np.nanmean(ious_aps)
        return miou, miou_aps


def evaluate():
    ## setup
    cfg = config_factory['resnet_cityscapes']
    args = parse_args()
    if not args.local_rank == -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
                    backend = 'nccl',
                    init_method = 'tcp://127.0.0.1:{}'.format(cfg.port),
                    world_size = torch.cuda.device_count(),
                    rank = args.local_rank
                    )
        setup_logger(cfg.respth)
    else:
        FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
        log_level = logging.INFO
        if dist.is_initialized() and dist.get_rank()!=0:
            log_level = logging.ERROR
        logging.basicConfig(level=log_level, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model')
    net = Deeplab_v3plus(cfg)
    save_pth = osp.join(cfg.respth, 'model_final.pth')
    net.load_state_dict(torch.load(save_pth), strict=False)
    net.cuda()
    net.eval()
    if not args.local_rank == -1:
        net = nn.parallel.DistributedDataParallel(net,
                device_ids = [args.local_rank, ],
                output_device = args.local_rank
                )

    ## evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(cfg)
    mIOU = evaluator(net)
    logger.info('mIOU is: {:.6f}'.format(mIOU))


if __name__ == "__main__":
    evaluate()
