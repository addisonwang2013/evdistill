from modeling.backbone import resnet, xception, drn, mobilenet, xception_new

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    if backbone == 'resnet_6ch':
        return resnet.ResNet101_6chs(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, pretrained=False)
    elif backbone =='xception_new':
        return xception_new.Xception(BatchNorm, pretrained=True)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
