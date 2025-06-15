import numpy as np
import torch
from torch import nn
from torchvision import models
from torchsummary import summary
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self,num_classes,vgg16_caffe_path=None, pretrained=False):
        super(Unet,self).__init__()
        num_filters = 32
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg16_caffe_path))

        self.encoder, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        self.conv1 = nn.Sequential(self.encoder[0],
                                   nn.ReLU(inplace=True),
                                   self.encoder[2],
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(self.encoder[5],
                                   nn.ReLU(inplace=True),
                                   self.encoder[7],
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(self.encoder[10],
                                   nn.ReLU(inplace=True),
                                   self.encoder[12],
                                   nn.ReLU(inplace=True),
                                   self.encoder[14],
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(self.encoder[17],
                                   nn.ReLU(inplace=True),
                                   self.encoder[19],
                                   nn.ReLU(inplace=True),
                                   self.encoder[21],
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(self.encoder[24],
                                   nn.ReLU(inplace=True),
                                   self.encoder[26],
                                   nn.ReLU(inplace=True),
                                   self.encoder[28],
                                   nn.ReLU(inplace=True))


        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)

        # self.outc1 = OutConv(num_filters*2, num_classes)
        self.outc2 = OutConv(num_filters, num_classes)


    def forward(self,x):
        x1 = self.conv1(x)  #64, 512, 512
        x2 = self.conv2(self.pool(x1))  #128, 256, 256
        x3 = self.conv3(self.pool(x2))  #256, 128, 128
        x4 = self.conv4(self.pool(x3))  #512, 64, 64
        x5 = self.conv5(self.pool(x4))  #512, 32, 32
        center = self.center(self.pool(x5))  #256, 32, 32

        dec5 = self.dec5(torch.cat([center, x5], 1))  #256, 64, 64

        dec4 = self.dec4(torch.cat([dec5, x4], 1))    #256,128,128
        dec3 = self.dec3(torch.cat([dec4, x3], 1))    #64, 256, 256
        dec2 = self.dec2(torch.cat([dec3, x2], 1))    # 32,512,512
        dec1 = self.dec1(torch.cat([dec2, x1], 1))    # 32,512,512

        x_out = self.outc2(dec1)  #1, 512, 512
        return x_out


if __name__=='__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(2).cuda()
    summary(model, (3, 512, 512))
    print(model.state_dict().keys())
    saved_state_dict = torch.load("../pretrained_model/fcn8s_from_caffe.pth")
    print(saved_state_dict.keys())


