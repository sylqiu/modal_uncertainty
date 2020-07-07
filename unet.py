from unet_blocks import *
import torch.nn.functional as F


class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters, apply_last_layer=True):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input_dim, output_dim, pool=pool))

        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input_dim = output_dim + self.num_filters[i]
            output_dim = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input_dim, output_dim))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output_dim, num_classes, kernel_size=1)



    def forward(self, x, val):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            # print(x.shape)
            if i != len(self.contracting_path)-1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i-1])

        del blocks

        #Used for saving the activations and plotting
        if val:
            self.activation_maps.append(x)
        
        if self.apply_last_layer:
            x =  self.last_layer(x)

        return x


class UnetEncoder(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_filters):
        super(UnetEncoder, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input_dim, output_dim, pool=pool))


    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            # print(x.shape)
            if i != len(self.contracting_path)-1:
                blocks.append(x)

        return x, blocks


class ResUnetEncoder(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_filters, bn=False):
        super(ResUnetEncoder, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.contracting_path = nn.ModuleList()
        self.bn = bn

        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlockRes(input_dim, output_dim, pool=pool, bn=self.bn))


    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            # print(x.shape)
            if i != len(self.contracting_path)-1:
                blocks.append(x)

        return x, blocks



class UnetDecoder(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, num_classes, num_filters, apply_last_layer=True):
        super(UnetDecoder, self).__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer

        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        output_dim = self.num_filters[-1]
        for i in range(n, -1, -1):
            input_dim = output_dim + self.num_filters[i]
            output_dim = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input_dim, output_dim))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output_dim, num_classes, kernel_size=1)


    def forward(self, x, blocks, val=False):


        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i-1])

        #Used for saving the activations and plotting
        if val:
            self.activation_maps.append(x)
        
        if self.apply_last_layer:
            x =  self.last_layer(x)

        return x


class ResUnetDecoder(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, num_classes, num_filters, posterior_layer, posterior_dim, apply_last_layer=True, bn=False):
        super(ResUnetDecoder, self).__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.bn = bn

        self.upsampling_path = nn.ModuleList()

        self.n = len(self.num_filters) - 2

        if posterior_layer == -1:
            posterior_layer = self.n + 1

        output_dim = self.num_filters[-1]
        for i in range(self.n, -1, -1):
            if posterior_layer == i+1:
                output_dim += posterior_dim
            input_dim = output_dim + self.num_filters[i]
            output_dim = self.num_filters[i]
            self.upsampling_path.append(UpConvBlockSkipRes(input_dim, output_dim, bn=self.bn))

        if self.apply_last_layer:
            last_layer = []
            for j in range(3-1):
                if posterior_layer == 0 and j == 0:
                    input_dim = output_dim + posterior_dim
                else:
                    input_dim = output_dim

                last_layer.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1))
                if self.bn:
                    last_layer.append(nn.BatchNorm2d(output_dim))
                last_layer.append(nn.ReLU(inplace=True))
                

            last_layer.append(nn.Conv2d(output_dim, self.num_classes, kernel_size=3, padding=1, bias=False))
            self.last_layer = nn.Sequential(*last_layer)


        self.posterior_layer = self.n+1 - posterior_layer


    def forward(self, x, posterior_z, blocks):


        for i, up in enumerate(self.upsampling_path):
            if i == self.posterior_layer:
                posterior_z = F.interpolate(posterior_z, [x.shape[2], x.shape[3]])
                x = torch.cat([x, posterior_z], dim=1)
            x = up(x, blocks[-i-1])

        
        if self.apply_last_layer:
            if self.posterior_layer == self.n + 1:
                posterior_z = F.interpolate(posterior_z, [x.shape[2], x.shape[3]])
                x = torch.cat([x, posterior_z], dim=1)
            x =  self.last_layer(x)

        return x


class MultiScaleResUnetDecoder(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, num_classes, num_filters, apply_last_layer=True, bn=False):
        super(MultiScaleResUnetDecoder, self).__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.apply_last_layer = apply_last_layer
        self.bn = bn

        self.upsampling_path = nn.ModuleList()
        self.predictor_list = nn.ModuleList()

        n = len(self.num_filters) - 2
        output_dim = self.num_filters[-1]
        for i in range(n, -1, -1):
            
            input_dim = output_dim + self.num_filters[i]
            self.predictor_list.append(Predictor(output_dim, 1))
            output_dim = self.num_filters[i]
            self.upsampling_path.append(UpConvBlockSkipRes(input_dim, output_dim, bn=self.bn))
            
            

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output_dim, num_classes, kernel_size=1, bias=False)


    def forward(self, x, blocks):

        predictions = []
        for i, up in enumerate(self.upsampling_path):
            predictions.append(self.predictor_list[i](x))
            x = up(x, blocks[-i-1])

        
        if self.apply_last_layer:
            x =  self.last_layer(x)

        predictions.append(x)
        return x, predictions


class MultiScaleBCELoss(nn.Module):
    def __init__(self):
        super(MultiScaleBCELoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.weight = [0.2, 0.2, 0.3, 0.3]

    def forward(self, predictions, target):
        loss = 0
        assert len(self.weight) == len(predictions)
        for i, seg in enumerate(predictions):
            tmp = predictions[i]
            loss += self.weight[i] * self.loss(tmp, F.interpolate(target, [tmp.shape[2], tmp.shape[3]]))

        return loss







class EncoderSimp(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_filters, no_first_pool=True):
        super(EncoderSimp, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = self.num_filters[i]

            if i == 0 and no_first_pool:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input_dim, output_dim, pool=pool))


    def forward(self, x):
        size_list = []
        size_list.append(x.shape[2:])
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            # if i != len(self.contracting_path)-1:
            size_list.append(x.shape[2:])

        return x, size_list



class Decoder(nn.Module):

    def __init__(self, num_classes, num_filters, apply_last_layer=True):
        super(Decoder, self).__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer

        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        output_dim = self.num_filters[-1]

        for i in range(n, -1, -1):
            input_dim = output_dim
            output_dim = self.num_filters[i]
            self.upsampling_path.append(UpConvBlockSimp(input_dim, output_dim))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output_dim, num_classes, kernel_size=1)

        self.size_list = []

    def forward(self, x, val=False):


        for i, up in enumerate(self.upsampling_path):
            x = up(x, self.size_list[-i-1])

        #Used for saving the activations and plotting
        if val:
            self.activation_maps.append(x)
        
        if self.apply_last_layer:
            x =  self.last_layer(x)

        return x
