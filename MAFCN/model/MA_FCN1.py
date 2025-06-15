import torch
from torch import nn
from math import sqrt
import warnings
import torch.nn.functional as F
from functools import partial
from torchvision import models
from torch.autograd import Variable

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')
nonlinearity = partial(F.relu, inplace=True)
warnings.filterwarnings('ignore')

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, in_channels=3, batch_norm=True):
    layers = []
    tmp_layers = []
    in_channels = in_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.Sequential(*tmp_layers)]
            tmp_layers = []
            tmp_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                tmp_layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                tmp_layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

class VGG(nn.Module):

    def __init__(self, cfg, in_channels=3):
        super(VGG, self).__init__()
        self.layer1, self.layer2, self.layer3, self.layer4, self.layer5 = make_layers(
            cfgs[cfg], in_channels=in_channels)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # x = self.maxpool(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x

# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #    layer
# ================================================================
#             Conv2d-1         [-1, 64, 512, 512]           1,792     1.0
#        BatchNorm2d-2         [-1, 64, 512, 512]             128     1.1
#               ReLU-3         [-1, 64, 512, 512]               0     1.2
#             Conv2d-4         [-1, 64, 512, 512]          36,928     1.3
#        BatchNorm2d-5         [-1, 64, 512, 512]             128     1.4
#               ReLU-6         [-1, 64, 512, 512]               0     1.5
#          MaxPool2d-7         [-1, 64, 256, 256]               0     2.0
#             Conv2d-8        [-1, 128, 256, 256]          73,856     2.1
#       BatchNorm2d-9        [-1, 128, 256, 256]             256     2.2
#              ReLU-10        [-1, 128, 256, 256]               0     2.3
#            Conv2d-11        [-1, 128, 256, 256]         147,584     2.4
#       BatchNorm2d-12        [-1, 128, 256, 256]             256     2.5
#              ReLU-13       [-1, 128, 256, 256]               0     2.6
#         MaxPool2d-14        [-1, 128, 128, 128]               0     3.0
#            Conv2d-15        [-1, 256, 128, 128]         295,168     3.1
#       BatchNorm2d-16        [-1, 256, 128, 128]             512     3.2
#              ReLU-17        [-1, 256, 128, 128]               0     3.3
#            Conv2d-18        [-1, 256, 128, 128]         590,080     3.4
#       BatchNorm2d-19        [-1, 256, 128, 128]             512     3.5
#              ReLU-20        [-1, 256, 128, 128]               0     3.6
#            Conv2d-21        [-1, 256, 128, 128]         590,080     3.7
#       BatchNorm2d-22        [-1, 256, 128, 128]             512     3.8
#              ReLU-23        [-1, 256, 128, 128]               0     3.9
#         MaxPool2d-24          [-1, 256, 64, 64]               0     4.0
#            Conv2d-25          [-1, 512, 64, 64]       1,180,160     4.1
#       BatchNorm2d-26          [-1, 512, 64, 64]           1,024     4.2
#              ReLU-27          [-1, 512, 64, 64]               0     4.3
#            Conv2d-28          [-1, 512, 64, 64]       2,359,808     4.4
#       BatchNorm2d-29          [-1, 512, 64, 64]           1,024     4.5
#              ReLU-30          [-1, 512, 64, 64]               0     4.6
#            Conv2d-31          [-1, 512, 64, 64]       2,359,808     4.7
#       BatchNorm2d-32          [-1, 512, 64, 64]           1,024     4.8
#              ReLU-33          [-1, 512, 64, 64]               0     4.9
#         MaxPool2d-34          [-1, 512, 32, 32]               0     5.0
#            Conv2d-35          [-1, 512, 32, 32]       2,359,808     5.1
#       BatchNorm2d-36          [-1, 512, 32, 32]           1,024     5.2
#              ReLU-37          [-1, 512, 32, 32]               0     5.3
#            Conv2d-38          [-1, 512, 32, 32]       2,359,808     5.4
#       BatchNorm2d-39          [-1, 512, 32, 32]           1,024     5.5
#              ReLU-40          [-1, 512, 32, 32]               0     5.6
#            Conv2d-41          [-1, 512, 32, 32]       2,359,808     5.7
#       BatchNorm2d-42          [-1, 512, 32, 32]           1,024     5.8
#              ReLU-43          [-1, 512, 32, 32]               0     5.9

def conv3x3_bn_relu(in_features, out_features):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True)
    )

def conv1x1_bn_relu(in_features, out_features):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=1),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True)
    )

class _PAM(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PAM, self).__init__()
        reduceRate = 8
        self.conv_b = nn.Conv2d(in_channels, in_channels // reduceRate, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // reduceRate, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out

class _CAM(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_CAM, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out

def output_layer(in_features, class_num):
    return nn.Conv2d(in_features, class_num, kernel_size=1, stride=1)

class _DAHead(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PAM(inter_channels, **kwargs)
        self.cam = _CAM(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 3, padding=1, bias=False),
            norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 3, padding=1, bias=False),
            norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        fusion_out = self.out(feat_fusion)

        return fusion_out

class NetFB(nn.Module):
    def __init__(self, class_num=1):
        '''
        Attention mechanism net based on VGG16
        TAM: three kinds of Attention for feature merge.
        CMA: channels merge for decode and encode part.
        params: 27.675
        flops: 298.732

        params: 26.375
        flops: 276.310

        params: 22.453
        flops: 227.818
        '''
        super(NetFB, self).__init__()
        self.class_num = class_num
        self.features = VGG('D', in_channels=3)
        self.layer1, self.layer2, self.layer3, self.layer4, self.layer5 = self.features.layer1, \
                                                                               self.features.layer2, \
                                                                               self.features.layer3, \
                                                                               self.features.layer4, \
                                                                               self.features.layer5
        channels = [64, 128, 256, 512, 512]
        self.TAM = _DAHead(channels[4])
        self.out5 = output_layer(channels[4], class_num=self.class_num)

        # self.up4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv1x1_bn_relu(channels[4], channels[3]))

        self.decoder4 = nn.Sequential(
            # CMA(channels[3]),
            conv1x1_bn_relu(channels[3]*2, channels[3]),
            conv3x3_bn_relu(channels[3], channels[3]),)
        self.out4 = output_layer(channels[3], class_num=self.class_num)

        # self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv1x1_bn_relu(channels[3], channels[2])
        )
        self.decoder3 = nn.Sequential(
            # CMA(channels[2]),
            conv1x1_bn_relu(channels[2]*2, channels[2]),
            conv3x3_bn_relu(channels[2], channels[2]), )
        self.out3 = output_layer(channels[2], class_num=self.class_num)

        # self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv1x1_bn_relu(channels[2], channels[1])
        )
        self.decoder2 = nn.Sequential(
            # CMA(channels[1]),
            conv1x1_bn_relu(channels[1]*2, channels[1]),
            conv3x3_bn_relu(channels[1], channels[1]),)
        self.out2 = output_layer(channels[1], class_num=self.class_num)

        # self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv1x1_bn_relu(channels[1], channels[0])
        )
        self.decoder1 = nn.Sequential(
            # CMA(channels[0]),
            conv1x1_bn_relu(channels[0]*2, channels[0]),
            conv3x3_bn_relu(channels[0], channels[0]),)
        self.num_features = channels[0]
        self.AFC = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

        self._initialize_weights()
        self.features.load_state_dict(torch.load('../pretrained_model/vgg16bn-encoder.pkl'))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)

        tam = self.TAM(e5)
        out5 = nn.Upsample(scale_factor=16)(self.out5(tam))

        up4 = self.up4(tam)
        d4 = self.decoder4(torch.cat((up4, e4), dim=1))
        out4 = nn.Upsample(scale_factor=8)(self.out4(d4))

        up3 = self.up3(d4)
        d3 = self.decoder3(torch.cat((up3, e3), dim=1))
        out3 = nn.Upsample(scale_factor=4)(self.out3(d3))

        up2 = self.up2(d3)
        d2 = self.decoder2(torch.cat((up2, e2), dim=1))
        out2 = nn.Upsample(scale_factor=2)(self.out2(d2))

        up1 = self.up1(d2)
        d1 = self.decoder1(torch.cat((up1, e1),dim=1))

        out1 = self.AFC(d1)

        return out1, out2, out3, out4, out5

if __name__=='__main__':
    # from torchsummary import summary
    # model = NetFB(2).cuda()
    # summary(model, (3, 512, 512))
    # print(model.features.state_dict().keys())
    # print(torch.cuda.device_count())
    # available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    # print(available_gpus)
    # model1 = torch.load('/media/disk2/lmy/dg/pretrained_model/vgg16bn-encoder.pkl')
    # print(model1.keys())
    # model2 = VGG('D', in_channels=3).cuda()
    # summary(model2, (3, 512, 512))

    # model1 = torch.load('/media/disk2/lmy/dg/MAFCN/smallall1/Snap_MAFCN1_geo_color_adain_ibn/epoch50miu_895.pth')
    # print(model1.keys())
    # model2 = torch.load('/media/disk2/lmy/dg/MAFCN/smallall1/Snap_MAFCN1/epoch40miu_914.pth')
    # print(model2.keys())
    model1 = torch.load('/media/disk2/lmy/swin_tiny_patch4_window7_224.pth')
    torch.save(model1,'/media/disk2/lmy/swin_tiny_patch4_window7_224_1.pth', _use_new_zipfile_serialization=False)