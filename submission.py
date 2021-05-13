
from modeling.deeplab import *
from models.deeplabv3plus import Deeplab_v3plus
from event_aps_dataset_v2 import EventAPS_Dataset
from configs import config_factory
import matplotlib as mpl
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


# decode segmentation maps for each class
def decode_segmap(temp):
    colors = [  # [  0,   0,   0],
        [128, 64, 128],#road
        # [244, 35, 232],#sidewalk
        [70, 70, 70],#building
        # [102, 102, 156],#wall
        # [190, 153, 153],#fence
        # [153, 153, 153],#pole
        [250, 170, 30],#traffic light
        # [220, 220, 0],#trafiic sign
        # [107, 142, 35],  # vegetation dark green
        [152, 251, 152],  # terrain bright green
        # [0, 130, 180],#sky
        [220, 20, 60], #person
        # [255, 0, 0],
        [0, 0, 142], #car
        # [0, 0, 70],
        # [0, 60, 100],
        # [0, 80, 100],
        # [0, 0, 230],
        # [119, 11, 32],
    ]

    label_colours = dict(zip(range(6), colors))
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 6):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

cfg = config_factory['eventDistill_config']

ds = EventAPS_Dataset(cfg, mode='test')

dl = DataLoader(ds,
                batch_size = 1,
                shuffle = True,
                sampler = None,
                num_workers = cfg.n_workers,
                pin_memory = True,
                drop_last = True)

## APS model
net = DeepLab(backbone='resnet',num_classes=6)
net.load_state_dict(torch.load(cfg.apsseg_ckpt))
net.eval()
net.cuda()

# event model
eventNet = DeepLab(backbone='resnet', num_classes=6)
eventNet.load_state_dict(torch.load(cfg.eventseg_ckpt))
eventNet.eval()
eventNet.cuda()


if not os.path.exists(cfg.save_dir):
    os.makedirs(cfg.save_dir)

with torch.no_grad():

    for index, (events, images, labels, name) in enumerate(dl):
        # visualization for APS images
        output = net(images.cuda())[0]
        prob = F.softmax(output, 1)
        probs = prob.detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        preds = preds.squeeze(0)
        decoded = decode_segmap(preds)

        # Visualization for events
        output_ev = eventNet(events.cuda())[0]
        prob_ev = F.softmax(output_ev, 1)
        probs_ev = prob_ev.detach().cpu().numpy()
        preds_ev = np.argmax(probs_ev, axis=1)
        preds_ev = preds_ev.squeeze(0)
        decoded_ev = decode_segmap(preds_ev)

        # Visualize the label maps
        decoded_labels = decode_segmap(labels.squeeze(0).squeeze(0).cpu().numpy())
        h,w,_ = np.shape(decoded)

        # make images
        new_image = np.zeros((h,5*w,3))
        new_image[:, 0:w, :] = events.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        new_image[:,w:2*w,:] = decoded_ev
        new_image[:,2*w:3*w,:] = images.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        new_image[:,3*w:4*w,:] = decoded
        new_image[:,4*w:5*w,:] = decoded_labels

        decoded_final = np.concatenate((images.squeeze(0).permute(1,2,0).detach().cpu().numpy(),decoded),axis=0)

        plt.imshow((new_image * 255).astype('uint8'))
        plt.axis('off')
        mpl.rcParams['agg.path.chunksize'] = 10000  # the default is 0
        plt.savefig(os.path.join(cfg.save_dir, "%s" % name[0]), dpi=600, bbox_inches='tight', pad_inches=0)
        print("saving image %d" % index)
        plt.clf()
