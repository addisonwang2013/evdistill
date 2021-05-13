import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder, build_aps_decoder
from modeling.backbone import build_backbone
from modeling.im_recon_decoder import build_im_recon_decoder
import math


# class DeepLab_Dual(nn.Module):
#     def __init__(self, backbone='resnet', output_stride=16, num_classes=6,
#                  sync_bn=True, freeze_bn=False, aux_loss=False, no_dropout= False, n_upsample=2, n_res=4, dim=256, output_dim=3):
#         super(DeepLab_Dual, self).__init__()
#         self.aux_loss = aux_loss
#         if backbone == 'drn':
#             output_stride = 8
#
#         if sync_bn == True:
#             BatchNorm = SynchronizedBatchNorm2d
#         else:
#             BatchNorm = nn.BatchNorm2d
#
#         self.backbone = build_backbone(backbone, output_stride, BatchNorm)
#         self.aspp = build_aspp(backbone, output_stride, BatchNorm, no_dropout= no_dropout)
#         self.decoder = build_decoder(num_classes, backbone, BatchNorm, no_dropout= no_dropout)
#         self.decoder2 = build_aps_decoder(n_upsample, n_res, dim, output_dim, backbone)
#
#         if self.aux_loss:
#             self.aux_logit = nn.Conv2d(256, num_classes, 1, 1, 0, bias=False)
#
#         self.freeze_bn = freeze_bn
#
#     def forward(self, input):
#         x_backdone, low_level_feat = self.backbone(input)
#         x_aspp = self.aspp(x_backdone)
#
#         if self.aux_loss:
#             x_aux = F.interpolate(self.aux_logit(x_aspp), size=input.size()[2:], mode='bilinear', align_corners=True)
#
#         x_decoder = self.decoder(x_aspp, low_level_feat)
#
#         x_aps_decoder = self.decoder2(low_level_feat)
#
#         x = F.interpolate(x_decoder, size=input.size()[2:], mode='bilinear', align_corners=True)
#
#         if self.aux_loss:
#             return x,  x_decoder, x_aps_decoder, x_aux
#         else:
#             return x, x_decoder, x_aps_decoder
#
#     def freeze_bn(self):
#         for m in self.modules():
#             if isinstance(m, SynchronizedBatchNorm2d):
#                 m.eval()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.eval()
#
#     def get_1x_lr_params(self):
#         modules = [self.backbone]
#         for i in range(len(modules)):
#             for m in modules[i].named_modules():
#                 if self.freeze_bn:
#                     if isinstance(m[1], nn.Conv2d):
#                         for p in m[1].parameters():
#                             if p.requires_grad:
#                                 yield p
#                 else:
#                     if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
#                             or isinstance(m[1], nn.BatchNorm2d):
#                         for p in m[1].parameters():
#                             if p.requires_grad:
#                                 yield p
#
#     def get_10x_lr_params(self):
#         modules = [self.aspp, self.decoder]
#         for i in range(len(modules)):
#             for m in modules[i].named_modules():
#                 if self.freeze_bn:
#                     if isinstance(m[1], nn.Conv2d):
#                         for p in m[1].parameters():
#                             if p.requires_grad:
#                                 yield p
#                 else:
#                     if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
#                             or isinstance(m[1], nn.BatchNorm2d):
#                         for p in m[1].parameters():
#                             if p.requires_grad:
#                                 yield p


#  image reconstruction ecoder
class EDSRConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EDSRConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )

        self.residual_upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            )

    def forward(self, input):
        return self.conv(input)+self.residual_upsampler(input)




class DeepLab_Dual(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab_Dual, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.sr_decoder = build_im_recon_decoder(num_classes, backbone, BatchNorm)

        # used to compute the affinity matrix
        self.pointwise = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes, 3, 1),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(inplace=True)
        )

        # self.upconv1 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
        # self.upconv2 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
        #
        # self.seg_conv_last = torch.nn.Sequential(nn.Conv2d(64, 64, 1),
        #                                         nn.BatchNorm2d(64),
        #                                         nn.ReLU(),
        #                                         nn.Conv2d(64, num_classes, 1),
        #                                         # nn.Tanh()
        #                                         )

        self.up_sr_1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.up_edsr_1 = EDSRConv(64, 64)
        # self.pixel_shuffle = nn.PixelShuffle(2)
        self.up_sr_2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up_edsr_2 = EDSRConv(32, 32)
        # self.up_sr_3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        # self.up_edsr_3 = EDSRConv(32,32)
        self.up_conv_last = nn.Conv2d(32, 3, 1)
        self.relu = nn.LeakyReLU(0.2)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x_seg = self.decoder(x, low_level_feat)
        x_sr= self.sr_decoder(x, low_level_feat)

        # #  simu
        # x_seg_up = self.relu(self.pixel_shuffle(self.upconv1(x_seg)))
        # x_seg_up = self.relu(self.pixel_shuffle(self.upconv2(x_seg_up)))
        # x_seg_up = self.seg_conv_last(x_seg_up)

        x_seg_up = F.interpolate(x_seg, size=input.size()[2:], mode='bilinear', align_corners=True)

        # x_seg_up = F.interpolate(x_seg_up,size=[2*i for i in input.size()[2:]], mode='bilinear', align_corners=True)
        # x_sr_0 = F.interpolate(x_sr, size=input.size()[2:], mode='bilinear', align_corners=True)

        x_sr_up = self.up_sr_1(x_sr)
        x_sr_up = self.up_edsr_1(x_sr_up)

        # x_sr_up= F.interpolate(x_sr, scale_factor=2, mode='nearest')
        # x_sr_up= self.relu(self.pixel_shuffle(x_sr_up))  # upsample with 2
        x_sr_up =self.up_sr_2(x_sr_up)
        x_sr_up = self.up_edsr_2(x_sr_up)
        # x_sr_up= self.relu(self.pixel_shuffle(x_sr_up))
        # x_sr_up = F.interpolate(x_sr_up, scale_factor=2, mode='nearest')

        # x_sr_up = self.up_sr_3(x_sr_up)
        # x_sr_up=self.up_edsr_3(x_sr_up)

        x_sr_up= self.up_conv_last(x_sr_up) #reconstruct images

        return x_seg_up, x_sr_up, self.pointwise(x_seg_up), x_sr_up

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


'''modify the image reconstruction branch and add batch normalization and relu layer in EIT branch'''
class DeepLab_Dual_V2(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab_Dual_V2, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        # self.aspp_eit =build_aspp(backbone, output_stride, BatchNorm, eit=True)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.sr_decoder = build_im_recon_decoder(num_classes, backbone, BatchNorm)

        # used to compute the affinity matrix
        self.pointwise = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes, 3, 1),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(inplace=True)
        )

        # self.upconv1 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
        # self.upconv2 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
        #
        # self.seg_conv_last = torch.nn.Sequential(nn.Conv2d(64, 64, 1),
        #                                         nn.BatchNorm2d(64),
        #                                         nn.ReLU(),
        #                                         nn.Conv2d(64, num_classes, 1),
        #                                         # nn.Tanh()
        #                                         )

        self.up_sr_1 = nn.ConvTranspose2d(64, 64, 2, stride=2)

        self.up_edsr_1 = EDSRConv(64, 64)

        # self.up_conv_1 = nn.Conv2d(64, 32, 3,1,1)
        # self.pixel_shuffle = nn.PixelShuffle(2)
        self.up_sr_2 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.up_edsr_2 = EDSRConv(32, 32)


        # self.up_conv_2 = nn.Conv2d(32,32, 3,1,1)
        # self.up_sr_3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        # self.up_edsr_3 = EDSRConv(32,32)
        self.up_conv_last = nn.Conv2d(32, 3, 1)
        self.tanh = nn.Tanh()
                                          # nn.Conv2d(32, 32, 3, 1, 1, bias=True),
                                          # nn.BatchNorm2d(32),
                                          # nn.ReLU(),

                                          # nn.Tanh()

        self.relu = nn.LeakyReLU(0.2)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        # x_sr = self.aspp_eit(x)

        x_seg = self.decoder(x, low_level_feat)
        x_sr= self.sr_decoder(x, low_level_feat)

        # #  simu
        # x_seg_up = self.relu(self.pixel_shuffle(self.upconv1(x_seg)))
        # x_seg_up = self.relu(self.pixel_shuffle(self.upconv2(x_seg_up)))
        # x_seg_up = self.seg_conv_last(x_seg_up)

        x_seg_up = F.interpolate(x_seg, size=input.size()[2:], mode='bilinear', align_corners=True)

        # x_seg_up = F.interpolate(x_seg_up,size=[2*i for i in input.size()[2:]], mode='bilinear', align_corners=True)
        # x_sr_0 = F.interpolate(x_sr, size=input.size()[2:], mode='bilinear', align_corners=True)

        x_sr_up = self.up_sr_1(x_sr)
        x_sr_up = self.up_edsr_1(x_sr_up)

        # x_sr_up= F.interpolate(x_sr, scale_factor=2, mode='nearest')
        # x_sr_up= self.relu(self.pixel_shuffle(x_sr_up))  # upsample with 2
        x_sr_up = self.up_sr_2(x_sr_up)
        x_sr_up = self.up_edsr_2(x_sr_up)
        # x_sr_up= self.relu(self.pixel_shuffle(x_sr_up))

        # x_sr_up = self.up_sr_3(x_sr_up)
        # x_sr_up=self.up_edsr_3(x_sr_up)

        # x_sr_up= self.up_conv_last(x_sr_up) #reconstruct images
        x_sr_up = self.tanh(self.up_conv_last(x_sr_up))  # reconstruct images

        return x_seg_up, x_sr_up, self.pointwise(x_seg_up), x_sr_up

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


class DeepLab_Dual_V3(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab_Dual_V3, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        # self.aspp_eit =build_aspp(backbone, output_stride, BatchNorm, eit=True)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.sr_decoder = build_im_recon_decoder(num_classes, backbone, BatchNorm)

        # used to compute the affinity matrix
        self.pointwise = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes, 3, 1),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(inplace=True)
        )

        # self.upconv1 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
        # self.upconv2 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
        #
        # self.seg_conv_last = torch.nn.Sequential(nn.Conv2d(64, 64, 1),
        #                                         nn.BatchNorm2d(64),
        #                                         nn.ReLU(),
        #                                         nn.Conv2d(64, num_classes, 1),
        #                                         # nn.Tanh()
        #                                         )

        # self.up_sr_1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        # self.up_edsr_1 = EDSRConv(64, 64)
        #
        # # self.up_conv_1 = nn.Conv2d(64, 32, 3,1,1)
        # # self.pixel_shuffle = nn.PixelShuffle(2)
        # self.up_sr_2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up_res = EDSRConv(64, 64)
        # self.up_conv_2 = nn.Conv2d(32,32, 3,1,1)
        # self.up_sr_3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        # self.up_edsr_3 = EDSRConv(32,32)
        self.upsampler = Upsample(4, 64)
        self.up_conv_last = nn.Sequential(
                                          # nn.Conv2d(32, 32, 3, 1, 1, bias=True),
                                          # nn.BatchNorm2d(32),
                                          # nn.ReLU(),
                                          nn.Conv2d(64, 3, 1),
                                          nn.Tanh()
                                          # nn.Tanh()
        )
        self.relu = nn.LeakyReLU(0.2)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        # x_sr = self.aspp_eit(x)

        x_seg = self.decoder(x, low_level_feat)
        x_sr= self.sr_decoder(x, low_level_feat)


        x_seg_up = F.interpolate(x_seg, size=input.size()[2:], mode='bilinear', align_corners=True)

        # x_seg_up = F.interpolate(x_seg_up,size=[2*i for i in input.size()[2:]], mode='bilinear', align_corners=True)
        # x_sr_0 = F.interpolate(x_sr, size=input.size()[2:], mode='bilinear', align_corners=True)

        x_sr_res= self.up_res(x_sr)
        x_sr_up = self.upsampler(x_sr_res)
        # x_sr_up = self.up_sr_3(x_sr_up)
        # x_sr_up=self.up_edsr_3(x_sr_up)

        x_sr_up= self.up_conv_last(x_sr_up) #reconstruct images

        return x_seg_up, x_sr_up, self.pointwise(x_seg_up), x_sr_up

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p



# use to predict the generated images from the task networks
class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=6,
                 sync_bn=True, freeze_bn=False, aux_loss=False, no_dropout=False):
        super(DeepLab, self).__init__()
        self.aux_loss = aux_loss
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm, no_dropout=no_dropout)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm, no_dropout=no_dropout)

        if self.aux_loss:
            self.aux_logit = nn.Conv2d(256, num_classes, 1, 1, 0, bias=False)

        self.freeze_bn = freeze_bn

    def forward(self, input):

        x_backdone, low_level_feat = self.backbone(input)
        x_aspp = self.aspp(x_backdone)

        if self.aux_loss:
            x_aux = F.interpolate(self.aux_logit(x_aspp), size=input.size()[2:], mode='bilinear', align_corners=True)

        x_decoder = self.decoder(x_aspp, low_level_feat)
        x = F.interpolate(x_decoder, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x, x


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())

