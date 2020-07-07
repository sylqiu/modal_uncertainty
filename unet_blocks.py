import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """
    def __init__(self, input_dim, output_dim, num_conv_layers=3, pool=True):
        super(DownConvBlock, self).__init__()
        layers = []

        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)) 
        for _ in range(num_conv_layers-1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1))  

        self.layers = nn.Sequential(*layers)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, patch):
        return self.activation(self.layers(patch))


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, num_conv_layers=2):
        super(Predictor, self).__init__()
        layers = []
        layers.append(nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1))
        for _ in range(num_conv_layers-2):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))  
        self.layers = nn.Sequential(*layers)
        self.activation = nn.ReLU(inplace=True)
        self.last_layer = nn.Conv2d(32, output_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.last_layer(self.activation(self.layers(x)))
        






class DownConvBlockRes(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """
    def __init__(self, input_dim, output_dim,  num_conv_layers=3, pool=True, bn=False):
        super(DownConvBlockRes, self).__init__()
        layers = []
        # print('{},{}'.format(input_dim, output_dim))
        bias = not bn
        self.shortcut = nn.Sequential()
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
                                        nn.Conv2d(input_dim, output_dim, kernel_size=1, padding=0, bias=bias))
        elif input_dim != output_dim:
            self.shortcut =  nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, padding=0, bias=bias))  

         
        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=bias)) 
        
        for _ in range(num_conv_layers-1):
            if bn:
                layers.append(nn.BatchNorm2d(output_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=bias))          

        self.layers = nn.Sequential(*layers)
        activation = []
        if bn:
            activation.append(nn.BatchNorm2d(output_dim))
        activation.append(nn.ReLU(inplace=True))
        self.activation = nn.Sequential(*activation)

    def forward(self, patch):
        return self.activation(self.shortcut(patch) + self.layers(patch))



class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim):
        super(UpConvBlock, self).__init__()
        self.conv_block = DownConvBlock(input_dim, output_dim, pool=False)

    def forward(self, x, bridge):

        up = nn.functional.interpolate(x, mode='bilinear', size=[bridge.shape[2], bridge.shape[3]], align_corners=True)
        out = torch.cat([up, bridge], 1)
        out =  self.conv_block(out)

        return out


class UpConvBlockSkipRes(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim, bn=False):
        super(UpConvBlockSkipRes, self).__init__()
        self.bn = bn
        self.conv_block = DownConvBlockRes(input_dim, output_dim, pool=False, bn=self.bn)

    def forward(self, x, bridge):

        up = nn.functional.interpolate(x, mode='bilinear', size=[bridge.shape[2], bridge.shape[3]], align_corners=True)
        out = torch.cat([up, bridge], 1)
        out =  self.conv_block(out)

        return out


class UpConvBlockRes(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim, bn=False):
        super(UpConvBlockRes, self).__init__()
        self.bn = bn
        self.conv_block = DownConvBlockRes(input_dim, output_dim, pool=False, bn=self.bn)

    def forward(self, x, size_list):

        up = nn.functional.interpolate(x, mode='bilinear', size=size_list, align_corners=True)
        out =  self.conv_block(up)

        return out

class UpConvBlockSimp(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim):
        super(UpConvBlockSimp, self).__init__()


        self.conv_block = DownConvBlock(input_dim, output_dim, pool=False)

    def forward(self, x, size):

        up = nn.functional.interpolate(x, mode='bilinear', size=size, align_corners=True)
        # print(up.shape)
        out = torch.cat([up], 1)
        out =  self.conv_block(out)

        return out

